import argparse
import os
import time

import numpy as np
import yaml
import joblib
import mujoco
import mujoco.viewer
import onnxruntime
from pynput import keyboard

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_S2R = os.path.join(_ROOT, "sim2real")

ROBOT_CFG = os.path.join(_ROOT, "humanoidverse/config/robot/Semi_Taks_T1/Semi_Taks_T1_20dof.yaml")
SCENE = os.path.join(_ROOT, "humanoidverse/data/robots/Semi_Taks_T1/scene_Semi_Taks_T1.xml")
MOTION = os.path.join(_ROOT, "humanoidverse/data/motions/Semi_Taks_T1/v1/accad_all.pkl")


def expand_gains(table, dof_names):
    out = np.zeros(len(dof_names), dtype=np.float64)
    for i, name in enumerate(dof_names):
        hit = [k for k in table if k in name]
        if not hit:
            raise ValueError(f"no gain key matches joint {name}")
        out[i] = float(table[hit[0]])
    return out


def main():
    ap = argparse.ArgumentParser(description="Semi_Taks_T1 (20dof, fixed-base upper-body imitation) sim2sim")
    ap.add_argument("--model_path", default="models/falcon/Semi_Taks_T1.onnx")
    ap.add_argument("--motion_idx", type=int, default=0)
    args = ap.parse_args()

    rcfg = yaml.safe_load(open(ROBOT_CFG))["robot"]
    dof_names = list(rcfg["dof_names"])
    ND = len(dof_names)  # 20
    default = np.array([rcfg["init_state"]["default_joint_angles"][n] for n in dof_names], dtype=np.float64)
    kp = expand_gains(rcfg["control"]["stiffness"], dof_names)
    kd = expand_gains(rcfg["control"]["damping"], dof_names)
    eff = np.array(rcfg["dof_effort_limit_list"], dtype=np.float64) * float(rcfg.get("dof_effort_limit_scale", 1.0))
    action_scale = float(rcfg["control"]["action_scale"])

    # upper_body_imitation obs is assembled MANUALLY (not alphabetically) in
    # humanoidverse/envs/upper_body_imitation/upper_body_imitation.py:_compute_observations:
    #   [dof_pos - default, dof_vel * 0.05, ref_dof_pos - default, last_actions]  (history = 1)
    DOF_VEL_SCALE = 0.05

    motions = joblib.load(MOTION)
    keys = list(motions.keys())
    n_motions = len(keys)

    os.chdir(_S2R)
    onnx_path = os.path.abspath(args.model_path)
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    sim_dt = 0.005
    m.opt.timestep = sim_dt
    sess = onnxruntime.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    qadr = np.zeros(ND, dtype=int)
    vadr = np.zeros(ND, dtype=int)
    for i, n in enumerate(dof_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise ValueError(f"joint {n} not found in {SCENE}")
        qadr[i] = m.jnt_qposadr[jid]
        vadr[i] = m.jnt_dofadr[jid]
    act2dof = np.full(m.nu, -1, dtype=int)
    for a in range(m.nu):
        jid = int(m.actuator_trnid[a, 0])
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in dof_names:
            act2dof[a] = dof_names.index(jname)
    if (act2dof < 0).any():
        raise ValueError("some actuators do not map to a policy joint")
    eff_act = eff[act2dof]

    st = {'mi': args.motion_idx % n_motions, 't': 0.0, 'paused': False,
          'act': np.zeros((1, ND), dtype=np.float32), 'reset': True}

    def cur_motion():
        return motions[keys[st['mi']]]

    def ref_at(t):
        mo = cur_motion()
        dof = np.asarray(mo['dof'], dtype=np.float64)  # (T, ND), already in Semi dof order
        fps = float(np.asarray(mo['fps']))
        T = dof.shape[0]
        f = int(round(t * fps)) % T
        return dof[f]

    def write_dof(q):
        for i in range(ND):
            d.qpos[qadr[i]] = q[i]

    def place_init():
        ref = ref_at(0.0)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        write_dof(ref)
        mujoco.mj_forward(m, d)
        st['t'] = 0.0
        st['act'] = np.zeros((1, ND), dtype=np.float32)

    def build_frame(ref):
        dof_pos = d.qpos[qadr]
        dof_vel = d.qvel[vadr]
        frame = np.concatenate([
            (dof_pos - default),
            dof_vel * DOF_VEL_SCALE,
            (ref - default),
            st['act'][0].astype(np.float64),
        ]).reshape(1, -1)
        return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def policy_step(ref):
        frame = build_frame(ref)
        raw = np.nan_to_num(sess.run([out_name], {in_name: frame})[0], nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(raw, -100.0, 100.0)
        st['act'] = action.astype(np.float32)
        return default + action_scale * action[0]

    def announce(tag=''):
        mo = cur_motion()
        T = np.asarray(mo['dof']).shape[0]
        fps = float(np.asarray(mo['fps']))
        print(f"[{tag}] 动作 {st['mi']+1}/{n_motions}  '{keys[st['mi']]}'  时长~{T/fps:.1f}s  "
              f"{'(暂停)' if st['paused'] else '(播放)'}", flush=True)

    def on_press(key):
        if key == keyboard.Key.space:
            st['paused'] = not st['paused']; announce('空格')
        else:
            ch = getattr(key, 'char', None)
            if ch in ('n', 'N'):
                st['mi'] = (st['mi'] + 1) % n_motions; st['reset'] = True; announce('n 下一个')
            elif ch in ('p', 'P'):
                st['mi'] = (st['mi'] - 1) % n_motions; st['reset'] = True; announce('p 上一个')
            elif ch in ('r', 'R'):
                st['reset'] = True; announce('r 重播')

    decim = max(1, round((1.0 / 50.0) / sim_dt))  # 50Hz policy
    control_dt = decim * sim_dt
    sync_every = max(1, round(0.02 / sim_dt))

    print("[Semi_Taks_T1_sim2sim] 20dof 固定基座 上半身模仿。基座焊死在世界坐标 (fix_base_link)。\n"
          "  策略实时跟踪参考动作 (ref_dof_pos 来自重定向后的动作库)。\n"
          "  键盘: 空格=暂停/播放, n=下一个动作, p=上一个, r=重播当前。\n")
    announce('启动')

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    q_target = default.copy()
    try:
        with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
            step = 0
            t_next = time.perf_counter()
            while viewer.is_running():
                if st['reset']:
                    place_init()
                    q_target = ref_at(0.0).copy()
                    st['reset'] = False
                    step = 0
                    t_next = time.perf_counter()
                ref = ref_at(st['t'])
                if step % decim == 0:
                    q_target = policy_step(ref)
                tau = kp * (q_target - d.qpos[qadr]) + kd * (0.0 - d.qvel[vadr])
                d.ctrl[:] = np.clip(tau[act2dof], -eff_act, eff_act)
                mujoco.mj_step(m, d)

                if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                    print("[稳定性保护] 状态 NaN/Inf → 复位重播。", flush=True)
                    st['reset'] = True
                    continue

                if step % decim == 0 and not st['paused']:
                    st['t'] += control_dt

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
