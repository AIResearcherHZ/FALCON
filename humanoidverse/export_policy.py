"""Export a trained FALCON checkpoint (logs/.../model_*.pt) to the deployment formats
used by sim2sim / sim2real:

  * <name>.onnx  -- input "actor_obs" (1, actor_obs_dim*history), output "action" (1, num_dof)
  * <name>.pt    -- TorchScript with the identical I/O contract

This is fully self-contained and does NOT import IsaacGym or build the simulator:
the decoupled multi-actor policy is rebuilt from the run's saved config.yaml and the
checkpoint weights, then traced. Works on CPU.

Examples
--------
# G1: write next to the checkpoint AND drop a copy into sim2real/models/falcon/g1_29dof.onnx
python humanoidverse/export_policy.py \
    +checkpoint=logs/g1_29dof_falcon/<run>/model_10000.pt \
    +output=sim2real/models/falcon/g1_29dof.onnx

# T1
python humanoidverse/export_policy.py \
    +checkpoint=$(ls -t logs/t1_29dof_falcon/*/model_*.pt | head -1) \
    +output=sim2real/models/falcon/t1_29dof.onnx

# Minimal: only write to <run>/exported/
python humanoidverse/export_policy.py +checkpoint=logs/g1_29dof_falcon/<run>/model_10000.pt
"""
import argparse
import copy
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from loguru import logger

import humanoidverse.utils.config_utils  # noqa: F401  (registers OmegaConf resolvers: len/eval/if/...)

# Project-local imports (no IsaacGym needed).
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.modules.ppo_modules import PPOActor
from humanoidverse.utils.inference_helpers import (
    export_multi_agent_decouple_policy_as_onnx,
    export_multi_agent_decouple_policy_as_jit,
)


def _strip_plus(argv):
    """Allow both '+checkpoint=...'/'checkpoint=...' (hydra-style) and '--checkpoint ...'."""
    out = []
    for a in argv:
        if a.startswith("+"):
            a = a[1:]
        out.append(a)
    return out


def parse_args():
    import sys
    ap = argparse.ArgumentParser(description="Export a FALCON checkpoint to ONNX + TorchScript.")
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", default=None,
                    help="Path to model_<iter>.pt inside a training run directory.")
    ap.add_argument("--output", default=None,
                    help="Optional explicit .onnx output path (a sibling .pt is written too). "
                         "e.g. sim2real/models/falcon/g1_29dof.onnx")
    # Accept hydra-style 'key=value' tokens so the command mirrors train/eval.
    argv = _strip_plus(sys.argv[1:])
    kv = [a for a in argv if "=" in a and not a.startswith("-")]
    rest = [a for a in argv if a not in kv]
    ns = ap.parse_args(rest)
    for item in kv:
        k, v = item.split("=", 1)
        if k in ("checkpoint", "ckpt"):
            ns.checkpoint = v
        elif k == "output":
            ns.output = v
        else:
            logger.warning(f"Ignoring unknown argument: {item}")
    if not ns.checkpoint:
        ap.error("checkpoint is required, e.g. +checkpoint=logs/.../model_10000.pt")
    return ns


def find_config(checkpoint: Path) -> Path:
    for cand in (checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find config.yaml next to checkpoint: looked in "
        f"{checkpoint.parent} and {checkpoint.parent.parent}")


def build_actors(cfg, checkpoint_path):
    """Rebuild the decoupled per-body-part actors exactly as the training algo does."""
    acfg = cfg.algo.config
    body_keys = list(cfg.robot.body_keys)
    num_act = {
        "lower_body": int(cfg.robot.lower_body_actions_dim),
        "upper_body": int(cfg.robot.upper_body_actions_dim),
    }
    obs_dim_dict = cfg.robot.algo_obs_dim_dict

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "actor_model_state_dict" in ckpt, \
        "Checkpoint has no 'actor_model_state_dict' -- is this a decoupled multi-actor run?"

    actors = {}
    for k in body_keys:
        module_cfg = copy.deepcopy(getattr(acfg.module_dict, "actor_" + k))
        actor = PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_cfg,
            num_actions=num_act[k],
            init_noise_std=acfg.init_noise_std[k],  # per-key scalar (matches training)
        )
        actor.load_state_dict(ckpt["actor_model_state_dict"][k])
        actor.eval()
        actors[k] = actor
        logger.info(f"actor[{k}]: input_dim={actor.actor_module.input_dim} "
                    f"output_dim={actor.actor_module.output_dim}")
    return actors, body_keys


def verify_onnx(onnx_path, wrapper_inputs_dim, torch_ref_fn):
    """Confirm the exported ONNX matches the torch policy numerically."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logger.info(f"ONNX inputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
    logger.info(f"ONNX outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    x = np.random.randn(1, wrapper_inputs_dim).astype(np.float32)
    onnx_out = sess.run([out_name], {in_name: x})[0]
    torch_out = torch_ref_fn(torch.from_numpy(x)).detach().numpy()
    max_err = float(np.abs(onnx_out - torch_out).max())
    logger.info(f"max|torch - onnx| = {max_err:.3e}  (output shape {onnx_out.shape})")
    if max_err > 1e-3:
        raise RuntimeError(f"ONNX/torch mismatch too large: {max_err}")
    return max_err


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config_path = find_config(checkpoint)
    logger.info(f"Loading run config from {config_path}")
    cfg = OmegaConf.load(config_path)
    pre_process_config(cfg)

    actors, body_keys = build_actors(cfg, checkpoint)
    inference_model = {"actors": actors}

    # Exact ONNX input width = the actor MLP's first-layer input (obs_dim * history_length).
    in_dim = actors[body_keys[0]].actor_module.input_dim
    example_obs_dict = {"actor_obs": torch.zeros(1, in_dim, dtype=torch.float32)}

    # Torch reference for verification (same wrapper used by the exporters).
    from humanoidverse.utils.inference_helpers import PPOMADecoupleWrapper
    ref_wrapper = PPOMADecoupleWrapper(
        {k: copy.deepcopy(actors[k]) for k in body_keys}, body_keys).eval()

    # 1) Always write next to the checkpoint, under exported/.
    export_dir = str(checkpoint.parent / "exported")
    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    jit_name = checkpoint.name  # keep model_<iter>.pt name for the TorchScript file

    export_multi_agent_decouple_policy_as_onnx(
        inference_model, export_dir, onnx_name, example_obs_dict, body_keys)
    export_multi_agent_decouple_policy_as_jit(
        inference_model, export_dir, jit_name, example_obs_dict, body_keys)
    onnx_path = os.path.join(export_dir, onnx_name)
    jit_path = os.path.join(export_dir, jit_name)
    logger.success(f"Exported ONNX -> {onnx_path}")
    logger.success(f"Exported JIT  -> {jit_path}")

    verify_onnx(onnx_path, in_dim, ref_wrapper)

    # 2) Optional explicit destination (e.g. sim2real/models/falcon/g1_29dof.onnx).
    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, out)
        logger.success(f"Copied ONNX  -> {out}")
        pt_out = out.with_suffix(".pt")
        shutil.copyfile(jit_path, pt_out)
        logger.success(f"Copied JIT   -> {pt_out}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
