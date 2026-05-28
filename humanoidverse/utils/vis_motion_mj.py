from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import time

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

import mujoco
import mujoco.viewer

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _resolve_scene(config: DictConfig) -> str:
    if config.get("scene_xml", None):
        scene = config.scene_xml
    else:
        motion_asset = config.robot.motion.get("asset", None) if "motion" in config.robot else None
        asset_root = motion_asset.get("assetRoot", None) if motion_asset else None
        asset_file = motion_asset.get("assetFileName", None) if motion_asset else None
        if asset_root is None and "asset" in config.robot:
            asset_root = config.robot.asset.get("assetRoot", None)
        scene = None
        if asset_root:
            asset_dir = Path(asset_root)
            if not asset_dir.is_absolute():
                asset_dir = REPO_ROOT / asset_dir

            fixed_base = bool(config.robot.asset.get("fix_base_link", False)) if "asset" in config.robot else False

            candidates = []
            if asset_file:
                stem = Path(asset_file).stem
                candidates.append(asset_dir / f"scene_{stem}.xml")
            if fixed_base:
                patterns = ("scene*.xml",)
            else:
                patterns = ("scene*freebase*.xml", "scene*.xml")
            for cand in candidates:
                if cand.is_file():
                    scene = str(cand)
                    break
            if scene is None:
                for pattern in patterns:
                    hits = [p for p in sorted(asset_dir.glob(pattern))
                            if fixed_base or "freebase" in p.name or "scene_" in p.name]
                    if fixed_base:
                        hits = [p for p in hits if "freebase" not in p.name]
                    if hits:
                        scene = str(hits[0])
                        break
        if scene is None:
            scene = str(REPO_ROOT / "humanoidverse/data/robots/g1/scene_g1_29dof_freebase.xml")
    if not os.path.isabs(scene):
        scene = str(REPO_ROOT / scene)
    return scene


class _PlayerState:
    def __init__(self, n_motions: int, motion_idx: int = 0):
        self.n_motions = n_motions
        self.motion_idx = motion_idx % max(n_motions, 1)
        self.t = 0.0
        self.paused = False
        self.restart = False
        self.advance = 0

    def key_callback(self, keycode: int) -> None:
        if keycode == 32:
            self.paused = not self.paused
            logger.info("paused" if self.paused else "playing")
        elif keycode in (ord("n"), ord("N")):
            self.advance = +1
        elif keycode in (ord("p"), ord("P")):
            self.advance = -1
        elif keycode in (ord("r"), ord("R")):
            self.restart = True
            logger.info("restart current motion")


if not any(a.startswith("robot=") or a.startswith("+robot=") for a in sys.argv[1:]):
    sys.argv.append("+robot=Semi_Taks_T1/Semi_Taks_T1_20dof")


@hydra.main(version_base="1.1", config_path="../config", config_name="base")
def main(config: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    motion_idx_start = int(config.get("motion_idx", 0))
    speed = float(config.get("speed", 1.0))
    loop = bool(config.get("loop", True))
    max_motions = int(config.get("max_motions", 256))

    logger.info(f"Motion file: {config.robot.motion.motion_file}")

    probe = MotionLibRobot(config.robot.motion, num_envs=1, device=device)
    n_unique = int(probe._num_unique_motions)
    del probe
    n_load = min(n_unique, max_motions)
    logger.info(f"Unique motions in file: {n_unique}; loading {n_load} for playback")

    motion_lib = MotionLibRobot(config.robot.motion, num_envs=n_load, device=device)
    motion_lib.load_motions(random_sample=False, start_idx=0)

    motion_keys = list(motion_lib.curr_motion_keys)
    fps_per_motion = motion_lib._motion_fps.detach().cpu().numpy()
    length_per_motion = motion_lib._motion_lengths.detach().cpu().numpy()

    scene_xml = _resolve_scene(config)
    logger.info(f"MuJoCo scene: {scene_xml}")
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    n_dof_cfg = int(config.robot.dof_obs_size)
    has_freejoint = (model.nq == n_dof_cfg + 7)
    if not has_freejoint and model.nq != n_dof_cfg:
        logger.warning(
            f"model.nq={model.nq} matches neither fixed-base (={n_dof_cfg}) "
            f"nor freebase (={n_dof_cfg + 7}); verify the scene xml matches the DoF ordering"
        )
    logger.info(f"Detected {'freebase' if has_freejoint else 'fixed-base'} robot "
                f"(model.nq={model.nq}, dof={n_dof_cfg})")

    state = _PlayerState(n_motions=n_load, motion_idx=motion_idx_start)

    with mujoco.viewer.launch_passive(model, data, key_callback=state.key_callback) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])

        while viewer.is_running():
            mid = state.motion_idx
            fps = float(fps_per_motion[mid])
            motion_len = float(length_per_motion[mid])
            dt = 1.0 / max(fps, 1e-6)
            logger.info(
                f"[{mid + 1}/{n_load}] {motion_keys[mid]}  len={motion_len:.2f}s @ {fps:.1f} fps"
            )

            state.t = 0.0
            state.restart = False
            state.advance = 0
            motion_ids = torch.tensor([mid], device=device)

            while viewer.is_running():
                if state.advance != 0:
                    state.motion_idx = (state.motion_idx + state.advance) % n_load
                    break

                if state.restart:
                    state.t = 0.0
                    state.restart = False

                if not state.paused:
                    mt = torch.tensor([state.t], device=device, dtype=torch.float32)
                    res = motion_lib.get_motion_state(motion_ids, mt)
                    root_pos = res["root_pos"][0].detach().cpu().numpy()
                    root_rot_xyzw = res["root_rot"][0].detach().cpu().numpy()
                    dof_pos = res["dof_pos"][0].detach().cpu().numpy()

                    if has_freejoint:
                        data.qpos[:3] = root_pos
                        data.qpos[3:7] = _quat_xyzw_to_wxyz(root_rot_xyzw)
                        n_dof = min(dof_pos.shape[0], model.nq - 7)
                        data.qpos[7:7 + n_dof] = dof_pos[:n_dof]
                    else:
                        n_dof = min(dof_pos.shape[0], model.nq)
                        data.qpos[:n_dof] = dof_pos[:n_dof]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)

                viewer.sync()
                time.sleep(dt / max(speed, 1e-6))

                if not state.paused:
                    state.t += dt
                    if state.t >= motion_len:
                        if loop:
                            state.t = 0.0
                        else:
                            state.advance = +1


if __name__ == "__main__":
    main()
