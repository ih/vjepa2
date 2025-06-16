import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.jetbot_ac.dataset import JetbotACDataset
from app.vjepa_droid.transforms import make_transforms
from src.hub.backbones import vjepa2_vit_giant
from src.models.ac_predictor import vit_ac_predictor
from src.utils.logging import AverageMeter


def main():
    parser = argparse.ArgumentParser(description="Train AC predictor on Jetbot data")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="jetbot_ac_ckpt.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transforms(random_horizontal_flip=False, crop_size=args.crop_size)

    dataset = JetbotACDataset(
        args.csv_path,
        args.data_dir,
        frames_per_clip=args.frames_per_clip,
        frameskip=args.frameskip,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    encoder, _ = vjepa2_vit_giant(pretrained=True)
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    predictor = vit_ac_predictor(
        img_size=(args.crop_size, args.crop_size),
        patch_size=args.patch_size,
        num_frames=args.frames_per_clip,
        tubelet_size=args.tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        depth=6,
        num_heads=encoder.num_heads,
        action_embed_dim=1,
        use_extrinsics=False,
    ).to(device)

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokens_per_frame = (args.crop_size // args.patch_size) ** 2

    for epoch in range(args.epochs):
        loss_meter = AverageMeter()
        for clips, actions, states in loader:
            clips = clips.to(device)
            actions = actions.to(device)
            states = states.to(device)

            with torch.no_grad():
                h = encoder(clips)
                h = h.view(clips.size(0), args.frames_per_clip, -1, h.size(-1)).flatten(1, 2)

            z_pred = predictor(h[:, :-tokens_per_frame], actions, states[:-1])
            h_target = h[:, tokens_per_frame:]
            loss = F.l1_loss(z_pred, h_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        print(f"Epoch {epoch+1}: loss {loss_meter.avg:.4f}")

    torch.save({"predictor": predictor.state_dict()}, args.output)


if __name__ == "__main__":
    main()
