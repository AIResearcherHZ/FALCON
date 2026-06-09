import argparse
import os
import time

import numpy as np
import yaml
import mujoco
import mujoco.viewer
import onnxruntime
from pynput import keyboard

_HERE = os.path.dirname(os.path.abspath(__file__))
_S2R = os.path.join(os.path.dirname(_HERE), "sim2real")

SORTED_OBS = ['actions', 'base_ang_vel', 'command_ang_vel', 'command_base_height',
              'command_lin_vel', 'command_stand', 'command_waist_dofs', 'dof_pos',
              'dof_vel', 'projected_gravity', 'ref_upper_dof_pos']
REF_UPPER = np.array([0.361541, 0.061779, 0.054225, -0.268456, 0.0, 0.0, 0.0,
                      0.361545, -0.061777, -0.054253, -0.268459, 0.0, 0.0, 0.0], dtype=np.float64)
BASE_HEIGHT_INIT = 0.793


def quat_rotate_inverse(q, v):
    qw = q[:, 0:1]
    qvec = q[:, 1:]
    a = v * (2.0 * qw ** 2 - 1.0)
    b = np.cross(qvec, v) * qw * 2.0
    c = qvec * np.sum(qvec * v, axis=1, keepdims=True) * 2.0
    return a - b + c


def main():
    ap = argparse.ArgumentParser(description="G1 sim2sim, self-contained single file")
    ap.add_argument("--config", default=os.path.join(_S2R, "config/g1/g1_29dof_falcon.yaml"))
    ap.add_argument("--model_path", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.chdir(_S2R)
    onnx_path = os.path.abspath(args.model_path or cfg["model_path"])

    ND = 29
    default = np.array(cfg["DEFAULT_DOF_ANGLES"], dtype=np.float64)
    kp = np.array(cfg["MOTOR_KP"], dtype=np.float64)
    kd = np.array(cfg["MOTOR_KD"], dtype=np.float64)
    eff = np.array(cfg["motor_effort_limit_list"], dtype=np.float64)
    qlo = np.array(cfg["motor_pos_lower_limit_list"], dtype=np.float64)
    qhi = np.array(cfg["motor_pos_upper_limit_list"], dtype=np.float64)
    sc = cfg["obs_scales"]
    sim_dt = float(cfg["SIMULATE_DT"])
    viewer_dt = float(cfg["VIEWER_DT"])
    base_h0 = float(cfg.get("DESIRED_BASE_HEIGHT", 0.75))
    action_scale = 0.25

    n_upper = int(cfg.get("NUM_UPPER_BODY_JOINTS", 14))
    upper_idx = list(range(ND - n_upper, ND))
    offset = default.copy()
    offset[upper_idx] = REF_UPPER
    ref_upper = REF_UPPER.reshape(1, -1)

    m = mujoco.MjModel.from_xml_path(cfg["ROBOT_SCENE"])
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt
    sess = onnxruntime.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    qpos_stand = np.zeros(m.nq)
    qpos_stand[0:3] = [0.0, 0.0, BASE_HEIGHT_INIT]
    qpos_stand[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos_stand[7:7 + ND] = offset
    m.qpos0[:] = qpos_stand

    st = {'obs': None, 'act': np.zeros((1, ND), dtype=np.float32), 'reset': False}
    cmd = dict(lin=np.zeros((1, 2)), ang=np.zeros((1, 1)), stand=np.zeros((1, 1)),
               base_h=np.array([[base_h0]], dtype=float), waist=np.zeros((1, 3)))
    HIST = int(cfg["history_length_dict"]["actor_obs"])

    def place_stand():
        d.qpos[:] = qpos_stand
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        st['obs'] = None
        st['act'] = np.zeros((1, ND), dtype=np.float32)
        cmd['lin'][:] = 0.0
        cmd['ang'][:] = 0.0
        cmd['stand'][:] = 0.0
        cmd['waist'][:] = 0.0
        cmd['base_h'][0, 0] = base_h0

    def build_frame():
        q = d.qpos
        v = d.qvel
        quat = q[3:7].reshape(1, 4)
        comp = {
            'actions': st['act'] * sc['actions'],
            'base_ang_vel': v[3:6].reshape(1, 3) * sc['base_ang_vel'],
            'command_ang_vel': cmd['ang'] * sc['command_ang_vel'],
            'command_base_height': cmd['base_h'] * sc['command_base_height'],
            'command_lin_vel': cmd['lin'] * sc['command_lin_vel'],
            'command_stand': cmd['stand'] * sc['command_stand'],
            'command_waist_dofs': cmd['waist'] * sc['command_waist_dofs'],
            'dof_pos': (q[7:7 + ND].reshape(1, ND) - default) * sc['dof_pos'],
            'dof_vel': v[6:6 + ND].reshape(1, ND) * sc['dof_vel'],
            'projected_gravity': quat_rotate_inverse(quat, np.array([[0.0, 0.0, -1.0]])) * sc['projected_gravity'],
            'ref_upper_dof_pos': ref_upper * sc['ref_upper_dof_pos'],
        }
        return np.concatenate([comp[k] for k in SORTED_OBS], axis=1).astype(np.float32)

    def policy_step():
        frame = build_frame()
        if st['obs'] is None:
            st['obs'] = np.zeros((1, frame.shape[1] * HIST), dtype=np.float32)
        st['obs'] = np.concatenate([st['obs'][:, frame.shape[1]:], frame], axis=1)
        action = np.clip(sess.run([out_name], {in_name: st['obs']})[0], -100.0, 100.0)
        st['act'] = action.astype(np.float32)
        return np.clip(action[0] * action_scale + offset, qlo, qhi)

    ACTION_CN = {'fwd': '前进', 'back': '后退', 'sl': '左移', 'sr': '右移', 'tl': '左转', 'tr': '右转',
                 'stop': '停下站立', 'toggle': '切换站立/行走', 'waist_l': '腰左转', 'waist_r': '腰右转',
                 'h_up': '升高', 'h_dn': '降低', 'reset': '复位'}

    def do(action, key=''):
        if action in ('fwd', 'back', 'sl', 'sr', 'tl', 'tr'):
            cmd['stand'][0, 0] = 1.0                       # movement auto-enables walk mode
            if action == 'fwd':
                cmd['lin'][0, 0] = np.clip(cmd['lin'][0, 0] + 0.1, -1.0, 1.0)
            elif action == 'back':
                cmd['lin'][0, 0] = np.clip(cmd['lin'][0, 0] - 0.1, -1.0, 1.0)
            elif action == 'sl':
                cmd['lin'][0, 1] = np.clip(cmd['lin'][0, 1] + 0.1, -0.5, 0.5)
            elif action == 'sr':
                cmd['lin'][0, 1] = np.clip(cmd['lin'][0, 1] - 0.1, -0.5, 0.5)
            elif action == 'tl':
                cmd['ang'][0, 0] = np.clip(cmd['ang'][0, 0] + 0.1, -1.0, 1.0)
            elif action == 'tr':
                cmd['ang'][0, 0] = np.clip(cmd['ang'][0, 0] - 0.1, -1.0, 1.0)
        elif action == 'stop':
            cmd['lin'][:] = 0.0
            cmd['ang'][:] = 0.0
            cmd['stand'][0, 0] = 0.0
        elif action == 'toggle':
            cmd['stand'][0, 0] = 0.0 if cmd['stand'][0, 0] else 1.0
            if cmd['stand'][0, 0] == 0.0:
                cmd['lin'][:] = 0.0
                cmd['ang'][:] = 0.0
        elif action == 'waist_l':
            cmd['waist'][0, 0] -= 0.2
        elif action == 'waist_r':
            cmd['waist'][0, 0] += 0.2
        elif action == 'h_up':
            cmd['base_h'][0, 0] += 0.1
        elif action == 'h_dn':
            cmd['base_h'][0, 0] -= 0.1
        elif action == 'reset':
            st['reset'] = True
        mode = '行走' if cmd['stand'][0, 0] else '站立'
        print(f"[按键 {key}] {ACTION_CN.get(action, action)}  →  模式={mode}  "
              f"前进={cmd['lin'][0, 0]:+.2f}  侧移={cmd['lin'][0, 1]:+.2f}  转向={cmd['ang'][0, 0]:+.2f}  "
              f"腰={cmd['waist'][0, 0]:+.1f}  高度={cmd['base_h'][0, 0]:.2f}", flush=True)

    KEYMAP = {'w': 'fwd', '8': 'fwd', 's': 'back', '2': 'back',
              'a': 'sl', '7': 'sl', 'd': 'sr', '9': 'sr',
              'q': 'tl', '4': 'tl', 'e': 'tr', '6': 'tr',
              'z': 'stop', '5': 'stop',
              ',': 'waist_l', '.': 'waist_r', '-': 'h_dn', '=': 'h_up', 'r': 'reset'}

    def on_press(key):
        if key == keyboard.Key.up:
            do('fwd', '↑')
        elif key == keyboard.Key.down:
            do('back', '↓')
        elif key == keyboard.Key.left:
            do('tl', '←')
        elif key == keyboard.Key.right:
            do('tr', '→')
        elif key == keyboard.Key.space:
            do('toggle', '空格')
        else:
            ch = getattr(key, 'char', None)
            if ch and ch in KEYMAP:
                do(KEYMAP[ch], ch)

    decim = max(1, round((1.0 / float(cfg.get("rl_rate", 50))) / sim_dt))
    sync_every = max(1, round(viewer_dt / sim_dt))

    place_stand()
    q_target = offset.copy()

    print("[g1_sim2sim] 原生策略。pynput 全局键盘:焦点在 MuJoCo 窗口 或 终端 都能控制。\n"
          "  方向键: ↑/↓ 前进/后退, ←/→ 左转/右转\n"
          "  小键盘: 8/2 前后, 4/6 转向, 7/9 侧移, 5 停 (也可用 w/s/a/d/q/e)\n"
          "  空格=站立/行走切换, z=停下站立, ,/.=腰左右, -/=升降高度, r=复位\n"
          "  默认: 站立不动。每次按键都会在下方打印出来。\n")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
            step = 0
            t_next = time.perf_counter()
            while viewer.is_running():
                if st['reset']:
                    place_stand()
                    q_target = offset.copy()
                    st['reset'] = False
                    step = 0
                if step % decim == 0:
                    q_target = policy_step()
                tau = kp * (q_target - d.qpos[7:7 + ND]) + kd * (0.0 - d.qvel[6:6 + ND])
                d.ctrl[6:6 + ND] = np.clip(tau, -eff, eff)
                mujoco.mj_step(m, d)
                if step % sync_every == 0:
                    viewer.sync()
                step += 1
                t_next += sim_dt
                dt = t_next - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                else:
                    t_next = time.perf_counter()
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
