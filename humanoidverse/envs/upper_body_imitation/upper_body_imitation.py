from __future__ import annotations

from typing import Dict

import torch
from loguru import logger

from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.torch_utils import to_torch


class UpperBodyImitation(BaseTask):
    """Fixed-base upper-body motion imitation env.

    Designed for Semi_Taks_T1 (20 DoFs, no feet, base welded to world). The
    motion library provides reference DoFs each frame; the agent learns PD
    targets that minimise tracking error.
    """

    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_motion_lib()
        self.is_evaluating = False
        self.control_mode = "wbc"
        self.init_done = True

    def _setup_robot_body_indices(self):
        body_names = self.body_names
        cfg = self.config.robot

        self.feet_indices = torch.zeros(0, dtype=torch.long, device=self.device)
        self.knee_indices = torch.zeros(0, dtype=torch.long, device=self.device)
        self.penalized_contact_indices = torch.zeros(0, dtype=torch.long, device=self.device)
        self.termination_contact_indices = torch.zeros(0, dtype=torch.long, device=self.device)

        if cfg.has_upper_body_dof:
            self.upper_dof_names = list(cfg.upper_dof_names)
            self.lower_dof_names = list(cfg.lower_dof_names)
            self.upper_dof_indices = [self.dof_names.index(d) for d in self.upper_dof_names]
            self.lower_dof_indices = [self.dof_names.index(d) for d in self.lower_dof_names]
        if hasattr(cfg, "waist_dof_names"):
            self.waist_dof_indices = [self.dof_names.index(d) for d in cfg.waist_dof_names]
        if hasattr(cfg, "arm_dof_names"):
            self.arm_dof_indices = [self.dof_names.index(d) for d in cfg.arm_dof_names]
        if hasattr(cfg, "left_arm_dof_names"):
            self.left_arm_dof_indices = [self.dof_names.index(d) for d in cfg.left_arm_dof_names]
        if hasattr(cfg, "right_arm_dof_names"):
            self.right_arm_dof_indices = [self.dof_names.index(d) for d in cfg.right_arm_dof_names]

        if cfg.has_torso:
            self.torso_name = cfg.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def _init_buffers(self):
        super()._init_buffers()
        self.torques = torch.zeros(self.num_envs, self.dim_actions, device=self.device)
        self.p_gains = torch.zeros(self.dim_actions, device=self.device)
        self.d_gains = torch.zeros(self.dim_actions, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)

        self.default_dof_pos = torch.zeros(self.num_dofs, device=self.device)
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos[i] = self.config.robot.init_state.default_joint_angles[name]
            for key, kp in self.config.robot.control.stiffness.items():
                if key in name:
                    self.p_gains[i] = kp
                    self.d_gains[i] = self.config.robot.control.damping[key]
                    break
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.motion_times = torch.zeros(self.num_envs, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _init_motion_lib(self):
        self._motion_lib = MotionLibRobot(
            self.config.robot.motion, num_envs=self.num_envs, device=self.device,
        )
        self._motion_lib.load_motions(random_sample=False, start_idx=0)
        self.motion_lengths = self._motion_lib._motion_lengths.to(self.device)
        self.motion_fps = self._motion_lib._motion_fps.to(self.device)
        self._sample_new_motions(torch.arange(self.num_envs, device=self.device))
        logger.info(f"Loaded {self._motion_lib._num_unique_motions} motions into UpperBodyImitation env.")

    def _sample_new_motions(self, env_ids: torch.Tensor):
        n = env_ids.shape[0]
        new_ids = torch.randint(0, self._motion_lib._num_unique_motions, (n,), device=self.device)
        self.motion_ids[env_ids] = new_ids
        self.motion_times[env_ids] = torch.rand(n, device=self.device) * self.motion_lengths[new_ids]

    def _refresh_ref_dof_pos(self):
        res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times)
        self.ref_dof_pos[:] = res["dof_pos"]

    def reset_envs_idx(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self._sample_new_motions(env_ids)
        self._refresh_ref_dof_pos()

        self.simulator.dof_pos[env_ids] = self.ref_dof_pos[env_ids]
        self.simulator.dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        action_scale = self.config.robot.control.action_scale
        target = self.default_dof_pos + action_scale * actions
        return self.p_gains * (target - self.simulator.dof_pos) - self.d_gains * self.simulator.dof_vel

    def _compute_observations(self) -> Dict[str, torch.Tensor]:
        obs = torch.cat([
            self.simulator.dof_pos - self.default_dof_pos,
            self.simulator.dof_vel * 0.05,
            self.ref_dof_pos - self.default_dof_pos,
            self.last_actions,
        ], dim=-1)
        return {"actor_obs": obs, "critic_obs": obs}

    def _compute_reward(self) -> torch.Tensor:
        dof_err = self.simulator.dof_pos - self.ref_dof_pos
        track = torch.exp(-2.0 * (dof_err ** 2).mean(dim=-1))
        action_rate = ((self.actions - self.last_actions) ** 2).mean(dim=-1)
        torque = (self.torques ** 2).mean(dim=-1) * 1e-5
        return track - 0.01 * action_rate - torque

    def _check_termination(self) -> torch.Tensor:
        time_out = self.episode_length_buf >= self.max_episode_length
        self.time_out_buf[:] = time_out
        return time_out

    def step(self, actor_state):
        actions = actor_state["actions"] if isinstance(actor_state, dict) else actor_state
        actions = torch.clip(actions, -self.config.normalization.clip_actions, self.config.normalization.clip_actions)
        self.actions[:] = actions

        sub_steps = self.config.simulator.config.sim.control_decimation
        for _ in range(sub_steps):
            self.torques = self._compute_torques(self.actions)
            self.simulator.apply_torques_at_dof(self.torques)
            self.simulator.simulate_at_each_physics_step()
        self._refresh_sim_tensors()

        dt = self.dt
        self.episode_length_buf += 1
        self.motion_times += dt
        self.motion_times = torch.where(
            self.motion_times >= self.motion_lengths[self.motion_ids],
            torch.zeros_like(self.motion_times),
            self.motion_times,
        )
        self._refresh_ref_dof_pos()

        rewards = self._compute_reward()
        dones = self._check_termination()

        env_ids_done = dones.nonzero(as_tuple=False).flatten()
        if env_ids_done.numel() > 0:
            self.reset_envs_idx(env_ids_done)
            self._refresh_sim_tensors()

        obs_dict = self._compute_observations()
        for key in self.obs_buf_dict.keys():
            if key in obs_dict:
                self.obs_buf_dict[key] = obs_dict[key]

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.simulator.dof_vel

        extras = {"time_outs": self.time_out_buf.clone(), "log": {}}
        return self.obs_buf_dict, rewards, dones, extras

    def set_is_evaluating(self):
        self.is_evaluating = True
