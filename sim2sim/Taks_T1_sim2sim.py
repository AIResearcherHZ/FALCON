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
_ROOT = os.path.dirname(_HERE)
_S2R = os.path.join(_ROOT, "sim2real")

ROBOT_CFG = os.path.join(_ROOT, "humanoidverse/config/robot/Taks_T1/Taks_T1_32dof.yaml")
OBS_CFG = os.path.join(_ROOT, "humanoidverse/config/obs/dec_loco/Taks_T1_32dof_obs_diff_force_history_wolinvel_ma.yaml")
SCENE = os.path.join(_ROOT, "humanoidverse/data/robots/Taks_T1/scene_Taks_T1_freebase.xml")

# dec_loco actor_obs is assembled by sorting the obs keys alphabetically
# (humanoidverse/envs/legged_base_task/legged_robot_base_ma.py: obs_keys = sorted(obs_config)).
SORTED_OBS = ['actions', 'base_ang_vel', 'command_ang_vel', 'command_base_height',
              'command_lin_vel', 'command_stand', 'command_waist_dofs', 'dof_pos',
              'dof_vel', 'projected_gravity', 'ref_upper_dof_pos']

BASE_HEIGHT_INIT = 0.75
# Commanded upper-body reference pose (arms 14 + neck 3 = 17). Zeros = hold the default
# neutral pose while the legs walk. Edit to demo a specific arm pose.
N_UPPER = 17
REF_UPPER = np.zeros(N_UPPER, dtype=np.float64)


def quat_rotate_inverse(q, v):
    qw = q[:, 0:1]
    qvec = q[:, 1:]
    a = v * (2.0 * qw ** 2 - 1.0)
    b = np.cross(qvec, v) * qw * 2.0
    c = qvec * np.sum(qvec * v, axis=1, keepdims=True) * 2.0
    return a - b + c


def expand_gains(table, dof_names):
    """Group-keyed stiffness/damping -> per-joint, matching the env rule `if key in joint_name`."""
    out = np.zeros(len(dof_names), dtype=np.float64)
    for i, name in enumerate(dof_names):
        hit = [k for k in table if k in name]
        if not hit:
            raise ValueError(f"no gain key matches joint {name}")
        out[i] = float(table[hit[0]])
    return out


def main():
    ap = argparse.ArgumentParser(description="Taks_T1 (32dof) sim2sim, self-contained single file")
    ap.add_argument("--model_path", default="models/falcon/Taks_T1.onnx")
    args = ap.parse_args()

    rcfg = yaml.safe_load(open(ROBOT_CFG))["robot"]
    ocfg = yaml.safe_load(open(OBS_CFG))["obs"]

    dof_names = list(rcfg["dof_names"])
    ND = len(dof_names)  # 32
    default = np.array([rcfg["init_state"]["default_joint_angles"][n] for n in dof_names], dtype=np.float64)
    kp = expand_gains(rcfg["control"]["stiffness"], dof_names)
    kd = expand_gains(rcfg["control"]["damping"], dof_names)
    eff = np.array(rcfg["dof_effort_limit_list"], dtype=np.float64) * float(rcfg.get("dof_effort_limit_scale", 1.0))
    qlo = np.array(rcfg["dof_pos_lower_limit_list"], dtype=np.float64)
    qhi = np.array(rcfg["dof_pos_upper_limit_list"], dtype=np.float64)
    action_scale = float(rcfg["control"]["action_scale"])
    sc = {k: float(v) for k, v in ocfg["obs_scales"].items()}
    HIST = int(ocfg["history_length"]["actor_obs"])

    n_upper = int(rcfg["upper_body_actions_dim"])  # 17
    assert n_upper == N_UPPER, f"upper dim {n_upper} != REF_UPPER {N_UPPER}"
    upper_idx = list(range(ND - n_upper, ND))  # dec_loco upper body is the contiguous tail
    offset = default.copy()
    offset[upper_idx] = REF_UPPER
    ref_upper = REF_UPPER.reshape(1, -1)

    os.chdir(_S2R)
    onnx_path = os.path.abspath(args.model_path)
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    sim_dt = 0.005
    m.opt.timestep = sim_dt
    sess = onnxruntime.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Name-based mapping (robust to MuJoCo's scrambled actuator order vs the policy dof order).
    qadr = np.zeros(ND, dtype=int)
    vadr = np.zeros(ND, dtype=int)
    for i, n in enumerate(dof_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise ValueError(f"joint {n} not found in {SCENE}")
        qadr[i] = m.jnt_qposadr[jid]
        vadr[i] = m.jnt_dofadr[jid]
    act2dof = np.full(m.nu, -1, dtype=int)        # actuator index -> policy dof index
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in dof_names:
            act2dof[a] = dof_names.index(jname)
    if (act2dof < 0).any():
        raise ValueError("some actuators do not map to a policy joint")
    eff_act = eff[act2dof]

    base_qadr = 0   # free joint root qpos start (pos[0:3], quat[3:7])
    base_vadr = 0   # free joint root qvel start (lin[0:3], ang[3:6])

    qpos_stand = m.qpos0.copy()
    qpos_stand[base_qadr:base_qadr + 3] = [0.0, 0.0, BASE_HEIGHT_INIT]
    qpos_stand[base_qadr + 3:base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    for i in range(ND):
        qpos_stand[qadr[i]] = offset[i]

    st = {'obs': None, 'act': np.zeros((1, ND), dtype=np.float32), 'reset': False}
    base_h0 = float(BASE_HEIGHT_INIT)
    cmd = dict(lin=np.zeros((1, 2)), ang=np.zeros((1, 1)), stand=np.zeros((1, 1)),
               base_h=np.array([[base_h0]], dtype=float), waist=np.zeros((1, 3)))

    def read_dof_pos():
        return d.qpos[qadr]

    def read_dof_vel():
        return d.qvel[vadr]

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
        quat = d.qpos[base_qadr + 3:base_qadr + 7].reshape(1, 4)
        ang_vel = d.qvel[base_vadr + 3:base_vadr + 6].reshape(1, 3)
        comp = {
            'actions': st['act'] * sc['actions'],
            'base_ang_vel': ang_vel * sc['base_ang_vel'],
            'command_ang_vel': cmd['ang'] * sc['command_ang_vel'],
            'command_base_height': cmd['base_h'] * sc['command_base_height'],
            'command_lin_vel': cmd['lin'] * sc['command_lin_vel'],
            'command_stand': cmd['stand'] * sc['command_stand'],
            'command_waist_dofs': cmd['waist'] * sc['command_waist_dofs'],
            'dof_pos': (read_dof_pos().reshape(1, ND) - default) * sc['dof_pos'],
            'dof_vel': read_dof_vel().reshape(1, ND) * sc['dof_vel'],
            'projected_gravity': quat_rotate_inverse(quat, np.array([[0.0, 0.0, -1.0]])) * sc['projected_gravity'],
            'ref_upper_dof_pos': ref_upper * sc['ref_upper_dof_pos'],
        }
        return np.concatenate([comp[k] for k in SORTED_OBS], axis=1).astype(np.float32)

    def policy_step():
        frame = np.nan_to_num(build_frame(), nan=0.0, posinf=0.0, neginf=0.0)
        if st['obs'] is None:
            st['obs'] = np.zeros((1, frame.shape[1] * HIST), dtype=np.float32)
        st['obs'] = np.concatenate([st['obs'][:, frame.shape[1]:], frame], axis=1)
        raw = np.nan_to_num(sess.run([out_name], {in_name: st['obs']})[0], nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(raw, -100.0, 100.0)
        st['act'] = action.astype(np.float32)
        return np.clip(action[0] * action_scale + offset, qlo, qhi)

    ACTION_CN = {'fwd': '前进', 'back': '后退', 'sl': '左移', 'sr': '右移', 'tl': '左转', 'tr': '右转',
                 'stop': '停下站立', 'toggle': '切换站立/行走', 'waist_l': '腰左转', 'waist_r': '腰右转',
                 'h_up': '升高', 'h_dn': '降低', 'reset': '复位'}

    def do(action, key=''):
        if action in ('fwd', 'back', 'sl', 'sr', 'tl', 'tr'):
            cmd['stand'][0, 0] = 1.0
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

    decim = max(1, round((1.0 / 50.0) / sim_dt))
    sync_every = max(1, round(0.02 / sim_dt))
    fall_tilt = 0.5
    fall_persist = max(1, round(0.3 / sim_dt))

    def upright_score():
        quat = d.qpos[base_qadr + 3:base_qadr + 7].reshape(1, 4)
        return float(-quat_rotate_inverse(quat, np.array([[0.0, 0.0, -1.0]]))[0, 2])

    place_stand()
    q_target = offset.copy()
    fall_cnt = 0

    print("[Taks_T1_sim2sim] 32dof 全身解耦运动策略。pynput 全局键盘:焦点在 MuJoCo 窗口 或 终端 都能控制。\n"
          "  方向键: ↑/↓ 前进/后退, ←/→ 左转/右转\n"
          "  小键盘: 8/2 前后, 4/6 转向, 7/9 侧移, 5 停 (也可用 w/s/a/d/q/e)\n"
          "  空格=站立/行走切换, z=停下站立, ,/.=腰左右, -/=升降高度, r=复位\n"
          "  默认: 站立不动。脖子3关节跟踪零参考(平视)。每次按键都会在下方打印出来。\n")

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
                    fall_cnt = 0
                    t_next = time.perf_counter()
                if step % decim == 0:
                    q_target = policy_step()
                tau = kp * (q_target - read_dof_pos()) + kd * (0.0 - read_dof_vel())
                d.ctrl[:] = np.clip(tau[act2dof], -eff_act, eff_act)
                mujoco.mj_step(m, d)

                finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all()
                if not finite:
                    bad = "状态出现 NaN/Inf (non-finite state)"
                elif upright_score() < fall_tilt:
                    fall_cnt += 1
                    bad = (f"机器人持续倾倒 tipped over (upright={upright_score():.2f}<{fall_tilt:.2f}, "
                           f"{fall_cnt * sim_dt:.1f}s)") if fall_cnt >= fall_persist else None
                else:
                    fall_cnt = 0
                    bad = None
                if bad is not None:
                    print(f"[稳定性保护] {bad} @ t={d.time:.2f}s → 自动复位站立 (auto-reset). "
                          f"策略不够鲁棒时可加大训练域随机化或挑 ep_len 更高的 checkpoint。", flush=True)
                    place_stand()
                    q_target = offset.copy()
                    viewer.sync()
                    step = 0
                    fall_cnt = 0
                    t_next = time.perf_counter()
                    continue

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
