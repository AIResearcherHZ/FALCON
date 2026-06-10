"""Headless evaluation -> mp4. Renders a trained FALCON policy in IsaacGym with NO
display/viewer, using an offscreen camera sensor, and writes a small mp4.

This is the headless replacement for `eval_agent.py` (whose eval_overrides set
headless=False and open a viewer -> crashes on a server with no DISPLAY).

Example
-------
python humanoidverse/record_video.py \
    +checkpoint=logs/g1_29dof_falcon/<run>/model_10000.pt \
    +record_seconds=12 +vx=0.5

Output: <run>/renderings/<model>.mp4  (override with +output=path.mp4)
Keyboard-free; the motion is driven by a simple velocity command schedule
(+vx / +vy / +yaw, and +stand_seconds before walking starts).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from loguru import logger

import humanoidverse.utils.config_utils  # noqa: F401  (registers OmegaConf resolvers: len/eval/if/...)


def _strip_plus(argv):
    return [a[1:] if a.startswith("+") else a for a in argv]


def parse_args():
    ap = argparse.ArgumentParser(description="Headless IsaacGym policy -> mp4 recorder.")
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", default=None)
    ap.add_argument("--output", default=None, help="Output .mp4 path.")
    ap.add_argument("--record_seconds", type=float, default=10.0)
    ap.add_argument("--stand_seconds", type=float, default=1.5, help="Stand still before walking.")
    ap.add_argument("--vx", type=float, default=0.0, help="Forward velocity command (m/s); 0 = stand.")
    ap.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
    ap.add_argument("--yaw", type=float, default=0.0, help="Yaw-rate command (rad/s).")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--cam_dist", type=float, default=3.0, help="Chase-cam distance behind robot.")
    ap.add_argument("--cam_fov", type=float, default=45.0, help="Camera horizontal FOV (deg).")
    argv = _strip_plus(sys.argv[1:])
    kv = [a for a in argv if "=" in a and not a.startswith("-")]
    rest = [a for a in argv if a not in kv]
    ns = ap.parse_args(rest)
    cast = {"record_seconds": float, "stand_seconds": float, "vx": float, "vy": float,
            "yaw": float, "width": int, "height": int, "cam_dist": float, "cam_fov": float}
    for item in kv:
        k, v = item.split("=", 1)
        if k in ("checkpoint", "ckpt"):
            ns.checkpoint = v
        elif k == "output":
            ns.output = v
        elif k in cast:
            setattr(ns, k, cast[k](v))
        else:
            logger.warning(f"Ignoring unknown argument: {item}")
    if not ns.checkpoint:
        ap.error("checkpoint is required, e.g. +checkpoint=logs/.../model_10000.pt")
    return ns


def find_config(checkpoint: Path) -> Path:
    for cand in (checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config.yaml not found near {checkpoint}")


def build_config(checkpoint: Path):
    cfg = OmegaConf.load(find_config(checkpoint))
    # mirror eval_agent's eval_overrides merge (fixes commands, disables wandb, etc.)
    if OmegaConf.select(cfg, "eval_overrides") is not None:
        cfg = OmegaConf.merge(cfg, cfg.eval_overrides)
    # force headless offscreen-recording mode (single env, graphics on, no viewer)
    forced = {"headless": True, "num_envs": 1, "auto_load_latest": False, "use_wandb": False}
    for k, v in forced.items():
        OmegaConf.update(cfg, k, v, force_add=True)
        OmegaConf.update(cfg, f"env.config.{k}", v, force_add=True)
    OmegaConf.update(cfg, "env.config.headless_record", True, force_add=True)
    OmegaConf.update(cfg, "checkpoint", str(checkpoint), force_add=True)
    # keep the demo clean: no random pushes during recording (top-level; env.config
    # references it via ${domain_rand}, so don't overwrite the interpolation node).
    OmegaConf.update(cfg, "domain_rand.push_robots", False)
    # Continuous footage: disable the termination conditions that otherwise reset the
    # episode mid-clip (slight lean -> gravity, motion-end ~10s, low height, ...).
    # env.config.termination is a concrete node, so updating leaf keys is safe.
    for k in ["terminate_by_gravity", "terminate_by_low_height", "terminate_by_contact",
              "terminate_when_motion_end", "terminate_when_low_upper_dof_tracking",
              "terminate_when_close_to_dof_pos_limit", "terminate_when_close_to_dof_vel_limit",
              "terminate_when_close_to_torque_limit"]:
        OmegaConf.update(cfg, f"env.config.termination.{k}", False, force_add=True)
    return cfg


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = build_config(checkpoint)

    # IsaacGym must be imported before torch.
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
    import torch
    import cv2
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo

    pre_process_config(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = instantiate(cfg.env, device=device)
    algo: BaseAlgo = instantiate(cfg.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(checkpoint))

    # ---- offscreen camera on env 0 (graphics enabled via headless_record flag) ----
    sim = env.simulator.sim
    gym = env.simulator.gym
    env_ptr = env.simulator.envs[0]
    cam_props = gymapi.CameraProperties()
    cam_props.width = args.width
    cam_props.height = args.height
    cam_props.horizontal_fov = args.cam_fov
    cam_props.enable_tensors = False
    cam = gym.create_camera_sensor(env_ptr, cam_props)
    if cam is None or cam == -1:
        raise RuntimeError("Failed to create camera sensor (graphics not available?).")
    # Rear, slightly to the side and above; aim near the pelvis (Z-up). This framing
    # keeps the robot centred (target must be low -- a high target pushes it off-frame).
    cam_dir = np.array([-0.80, -0.30, 0.52], dtype=np.float64)  # unit vector
    cam_offset = cam_dir * args.cam_dist

    def grab_frame():
        base = env.simulator.robot_root_states[0, 0:3].detach().cpu().numpy().astype(np.float64)
        eye = base + cam_offset
        # aim at the ground point under the robot (fixed Z) -> robot stays centred & whole
        tgt = np.array([base[0], base[1], 0.35])
        gym.set_camera_location(cam, env_ptr, gymapi.Vec3(*eye), gymapi.Vec3(*tgt))
        # GPU pipeline: sync physics results to the graphics buffers before step_graphics,
        # otherwise the articulated-body transforms use zero joint angles and all limbs
        # collapse onto the base (robot renders as a limbless "barrel").
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env_ptr, cam, gymapi.IMAGE_COLOR)
        img = img.reshape(args.height, args.width, 4)[:, :, :3]  # RGBA -> RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---- evaluation rollout with a simple command schedule ----
    env.set_is_evaluating()
    obs = env.reset_all()
    policy = algo._get_inference_policy()

    control_dt = env.dt  # seconds per control step (0.02 -> 50Hz)
    fps = round(1.0 / control_dt)
    num_steps = int(args.record_seconds * fps)
    stand_steps = int(args.stand_seconds * fps)
    logger.info(f"Recording {num_steps} steps @ {fps} fps "
                f"({args.record_seconds}s), {args.width}x{args.height}")

    frames = []
    resets = 0
    for step in range(num_steps):
        walking = step >= stand_steps
        # commands: [vx, vy, yaw_rate(auto), heading, walk(0/1), ...,  base_height(8)]
        # Velocity commands only take effect when walk=1 AND tapping_in_place=1
        # (tapping_in_place=0 means step-in-place; it defaults to 0 in eval).
        env.tapping_in_place[:] = 1.0 if walking else 0.0
        env.commands[:, 0] = args.vx if walking else 0.0
        env.commands[:, 1] = args.vy if walking else 0.0
        env.commands[:, 4] = 1.0 if walking else 0.0
        if walking and args.yaw != 0.0:
            env.commands[:, 3] += args.yaw * control_dt  # accumulate target heading
        with torch.no_grad():
            action = policy(obs["actor_obs"])
        obs, _, reset_buf, _ = env.step({"actions": action})
        resets += int(reset_buf[0].item())
        frames.append(grab_frame())
        if step % 50 == 0:
            b = env.simulator.robot_root_states[0, 0:3].detach().cpu().tolist()
            logger.info(f"  step {step}/{num_steps}  base=({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f})  resets={resets}")

    # ---- write mp4 ----
    out = Path(args.output) if args.output else (checkpoint.parent / "renderings" /
                                                 f"{checkpoint.stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (args.width, args.height))
    for f in frames:
        writer.write(f)
    writer.release()

    mean_lum = float(np.mean([f.mean() for f in frames]))
    logger.success(f"Wrote {len(frames)} frames -> {out}  (mean pixel {mean_lum:.1f})")
    if mean_lum < 1.0:
        logger.warning("Frames look black -- offscreen GPU rendering may have failed.")


if __name__ == "__main__":
    main()
