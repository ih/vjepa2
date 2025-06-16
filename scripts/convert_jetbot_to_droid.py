import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Jetbot CSV to DROID dataset format")
    parser.add_argument("csv_path", type=str, help="Path to Jetbot CSV")
    parser.add_argument("data_dir", type=str, help="Directory containing Jetbot images")
    parser.add_argument("output_dir", type=str, help="Where to write DROID formatted dataset")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for generated videos")
    return parser.parse_args()


def read_csv(csv_path):
    sessions = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions[row["session_id"].strip()].append(row)
    # sort by timestamp within session
    for s in sessions.values():
        s.sort(key=lambda r: float(r["timestamp"]))
    return sessions


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def create_video(session_rows, base_dir, video_path, fps):
    # assume at least one row
    first_img = cv2.imread(os.path.join(base_dir, session_rows[0]["image_path"]))
    if first_img is None:
        raise FileNotFoundError(session_rows[0]["image_path"])
    height, width = first_img.shape[:2]
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for row in session_rows:
        img = cv2.imread(os.path.join(base_dir, row["image_path"]))
        if img is None:
            raise FileNotFoundError(row["image_path"])
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()


def build_trajectory(session_rows):
    n = len(session_rows)
    cartesian = np.zeros((n, 6), dtype=np.float32)
    gripper = np.zeros((n,), dtype=np.float32)
    # integrate 1D action into the first axis of cartesian position
    pos = 0.0
    for i, row in enumerate(session_rows):
        try:
            action = float(row["action"])
        except ValueError:
            action = 0.0
        pos += action
        cartesian[i, 0] = pos
    extrinsics = np.zeros((n, 6), dtype=np.float32)
    return cartesian, gripper, extrinsics


def write_h5(path, cartesian, gripper, extrinsics):
    with h5py.File(path, "w") as f:
        obs = f.create_group("observation")
        robot = obs.create_group("robot_state")
        robot.create_dataset("cartesian_position", data=cartesian)
        robot.create_dataset("gripper_position", data=gripper)
        ce = obs.create_group("camera_extrinsics")
        ce.create_dataset("left_left", data=extrinsics)


def process_sessions(sessions, base_dir, out_dir, fps):
    list_file = os.path.join(out_dir, "droid_paths.csv")
    lines = []
    for sid, rows in sessions.items():
        sdir = os.path.join(out_dir, sid)
        mp4_dir = os.path.join(sdir, "recordings", "MP4")
        ensure_dir(mp4_dir)
        video_path = os.path.join(mp4_dir, "left.mp4")
        create_video(rows, base_dir, video_path, fps)
        cartesian, gripper, extrinsics = build_trajectory(rows)
        write_h5(os.path.join(sdir, "trajectory.h5"), cartesian, gripper, extrinsics)
        meta = {"left_mp4_path": os.path.join("recordings", "MP4", "left.mp4")}
        with open(os.path.join(sdir, "metadata.json"), "w") as f:
            json.dump(meta, f)
        lines.append(sdir)
    with open(list_file, "w") as f:
        for l in lines:
            f.write(l + "\n")
    print(f"Wrote dataset list to {list_file}")


def main():
    args = parse_args()
    sessions = read_csv(args.csv_path)
    ensure_dir(args.output_dir)
    process_sessions(sessions, args.data_dir, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
