from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse

import joblib
import numpy as np

G1_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

T1_DOF_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
    "neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint",
]

T1_JOINT_AXES = np.array([
    [0, 0, 1], [1, 0, 0], [0, 1, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0],
    [1, 0, 0], [0, 0, 1], [0, 1, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0],
    [1, 0, 0], [0, 0, 1], [0, 1, 0],
    [0, 0, 1], [1, 0, 0], [0, 1, 0],
], dtype=np.float32)

T1_DOF_LIMITS = np.array([
    [-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52],
    [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.618, 2.618], [-0.7, 1.57],
    [-2.67, 2.67], [-0.9, 0.9], [-0.9, 0.9],
    [-3.0892, 2.6704], [-2.2515, 1.5882], [-2.618, 2.618], [-0.7, 1.57],
    [-2.67, 2.67], [-0.9, 0.9], [-0.9, 0.9],
    [-1.57, 1.57], [-0.873, 0.873], [-0.873, 0.873],
], dtype=np.float32)


def build_dof_index_map() -> np.ndarray:
    g1_idx = {n: i for i, n in enumerate(G1_DOF_NAMES)}
    mapping = np.full(len(T1_DOF_NAMES), -1, dtype=np.int64)
    for i, name in enumerate(T1_DOF_NAMES):
        if name in g1_idx:
            mapping[i] = g1_idx[name]
    return mapping


def retarget_one(g1_motion: dict, dof_map: np.ndarray) -> dict:
    g1_dof = np.asarray(g1_motion["dof"], dtype=np.float32)
    T = g1_dof.shape[0]

    t1_dof = np.zeros((T, len(T1_DOF_NAMES)), dtype=np.float32)
    for i, gi in enumerate(dof_map):
        if gi >= 0:
            t1_dof[:, i] = g1_dof[:, gi]

    np.clip(t1_dof, T1_DOF_LIMITS[:, 0], T1_DOF_LIMITS[:, 1], out=t1_dof)

    pose_aa = np.zeros((T, 1 + len(T1_DOF_NAMES), 3), dtype=np.float32)
    pose_aa[:, 1:, :] = t1_dof[:, :, None] * T1_JOINT_AXES[None, :, :]

    root_trans = np.zeros((T, 3), dtype=np.float32)
    root_rot = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, 1))

    out = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        "dof": t1_dof,
        "root_rot": root_rot,
        "fps": int(g1_motion.get("fps", 30)),
    }
    if "smpl_joints" in g1_motion:
        out["smpl_joints"] = np.asarray(g1_motion["smpl_joints"], dtype=np.float32)
    return out


def retarget_file(in_path: Path, out_path: Path) -> None:
    print(f"Loading {in_path}")
    g1_data = joblib.load(in_path)
    if not isinstance(g1_data, dict):
        raise ValueError(f"Expected dict at top level of {in_path}, got {type(g1_data)}")

    dof_map = build_dof_index_map()
    missing = [T1_DOF_NAMES[i] for i, gi in enumerate(dof_map) if gi < 0]
    print(f"Mapped {sum(dof_map >= 0)}/{len(T1_DOF_NAMES)} DoFs from G1. "
          f"Zero-filled (no G1 source): {missing}")

    out_data = {}
    for k, v in g1_data.items():
        try:
            out_data[k] = retarget_one(v, dof_map)
        except Exception as e:
            print(f"  skip {k}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(out_data)} motions -> {out_path}")
    joblib.dump(out_data, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        default=str(REPO_ROOT / "humanoidverse/data/motions/g1_29dof/v1/accad_all.pkl"),
    )
    ap.add_argument(
        "--out_path",
        default=str(REPO_ROOT / "humanoidverse/data/motions/Semi_Taks_T1/v1/accad_all.pkl"),
    )
    args = ap.parse_args()
    retarget_file(Path(args.in_path), Path(args.out_path))


if __name__ == "__main__":
    main()
