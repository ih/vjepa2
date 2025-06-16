import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class JetbotACDataset(Dataset):
    """Dataset of Jetbot sessions for training an action-conditioned predictor."""

    def __init__(self, csv_path, data_dir, frames_per_clip=8, frameskip=1, transform=None):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.transform = transform
        self.sessions = self._load_sessions()

    def _load_sessions(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        required = {"session_id", "image_path", "action"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        sessions = []
        for _, g in df.groupby("session_id"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            sessions.append(g)
        return sessions

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions[idx]
        needed = self.frames_per_clip * self.frameskip
        if len(sess) < needed:
            raise ValueError(f"Session length {len(sess)} < required {needed}")
        start = np.random.randint(0, len(sess) - needed + 1)
        indices = np.arange(start, start + needed, self.frameskip)
        frames, acts = [], []
        for i in indices:
            row = sess.iloc[i]
            img = Image.open(os.path.join(self.data_dir, row["image_path"])).convert("RGB")
            img = self.transform(img) if self.transform else T.ToTensor()(img)
            frames.append(img)
            acts.append(row["action"])
        frames = torch.stack(frames, dim=1)  # C T H W
        states = torch.tensor(acts, dtype=torch.float32).unsqueeze(-1)
        actions = states[1:] - states[:-1]
        return frames, actions, states
