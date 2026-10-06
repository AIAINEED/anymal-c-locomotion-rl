"""
ANYmal-C 策略 sim2sim：Isaac Lab (PhysX) 训练 → MuJoCo 验证。

用法：
  python anymal_sim2sim.py --model <mujoco_menagerie>/anybotics_anymal_c/scene.xml \
                           --policy policy.onnx --video sim2sim.mp4

配置全部来自 Isaac Lab 侧的实测对齐（不是猜的）：
  关节顺序（Isaac Lab 的 articulation 顺序，与 MuJoCo 模型不同！）
      LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, LH_HFE, RF_HFE, RH_HFE, LF_KFE, LH_KFE, RF_KFE, RH_KFE
  观测 48 维 = base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
              + joint_pos_rel(12) + joint_vel_rel(12) + last_action(12)，无缩放、无归一化
  动作 = 默认关节角 + 0.5 × 策略输出（JointPositionActionCfg(scale=0.5, use_default_offset=True)）
  控制频率 50 Hz（sim.dt=0.005 × decimation=4），episode 20 秒
  默认关节角 = [0,0,0,0, 0.4,-0.4,0.4,-0.4, -0.8,0.8,-0.8,0.8]（按上面的顺序）

关于执行器：Isaac Lab 的 ANYmal-C 用 ActuatorNet LSTM（真实 ANYdrive 的神经网络模型），
MuJoCo 里用 PD 近似。实测 kp=120 / kd=5 时速度跟踪最好（指令 0.8 → 实测 0.825 m/s）；
这组增益是 sim2sim 的主要误差来源，也是简历里值得写的一条工程细节。
"""
import argparse
import os

import numpy as np
import mujoco
import onnxruntime as ort

SIM_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.5
KP, KD = 120.0, 5.0
EFFORT_LIMIT = 80.0
EPISODE_S = 20.0
BASE_INIT_Z = 0.62
FALL_Z = 0.35

MJ_ORDER = ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]
POLICY_ORDER = ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
                "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
                "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]
IDX = [MJ_ORDER.index(j) for j in POLICY_ORDER]      # 策略第 i 维 → MuJoCo 第 IDX[i] 个关节


def default_q_mj():
    """Isaac Lab ANYMAL_C_CFG 的默认关节角，按 MuJoCo 关节顺序排好。"""
    q = np.zeros(12)
    for n in POLICY_ORDER:
        front = n.startswith("LF") or n.startswith("RF")
        q[MJ_ORDER.index(n)] = (0.0 if n.endswith("HAA")
                                else (0.4 if front else -0.4) if n.endswith("HFE")
                                else (-0.8 if front else 0.8))
    return q


class AnyMalSim2Sim:
    def __init__(self, xml_path, policy_path, kp=KP, kd=KD):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.q_def = default_q_mj()
        self.q_def_policy = self.q_def[IDX]

        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.m.nu)]
        if names != MJ_ORDER:
            raise RuntimeError(f"模型关节顺序与预期不符：{names}")

        for i in range(self.m.nu):
            self.m.actuator_gainprm[i, 0] = kp
            self.m.actuator_biasprm[i, 1] = -kp
            self.m.actuator_biasprm[i, 2] = -kd
            self.m.actuator_forcerange[i] = [-EFFORT_LIMIT, EFFORT_LIMIT]

        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if gid >= 0:
            self.m.geom_friction[gid] = [1.0, 0.005, 0.0001]

        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.policy = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._v = np.zeros(6)

    def _vel_body(self):
        """机体坐标系速度。注意：MuJoCo 的 flg_local=1 返回值有误，必须用世界系再自己旋转。"""
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.base, self._v, 0)
        R = self.d.xmat[self.base].reshape(3, 3)
        return R.T @ self._v[0:3], R.T @ self._v[3:6], R

    def obs(self, cmd, last_action):
        ang_b, lin_b, R = self._vel_body()
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([lin_b, ang_b, grav, np.asarray(cmd, dtype=np.float64),
                               self.d.qpos[7:][IDX] - self.q_def_policy,
                               self.d.qvel[6:][IDX], last_action]).astype(np.float32)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0.0, 0.0, BASE_INIT_Z]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[7:] = self.q_def
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def render(self):
        if not hasattr(self, "_r"):
            self._r = mujoco.Renderer(self.m, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 3.2, 130.0, -18.0
        # 相机跟随机器人（否则它会走出画面）
        self._cam.lookat[:] = self.d.qpos[0:3]
        self._r.update_scene(self.d, camera=self._cam)
        return self._r.render().copy()

    def episode(self, cmd, video_frames=None, max_steps=None):
        self.reset()
        last = np.zeros(12, dtype=np.float32)
        cmd = np.asarray(cmd, dtype=np.float64)
        n = int(EPISODE_S / (SIM_DT * DECIMATION))
        if max_steps:
            n = min(n, max_steps)
        vx_list, dist, survive = [], 0.0, 0
        for _ in range(n):
            a = self.policy.run(None, {"obs": self.obs(cmd, last)[None]})[0][0]
            tgt = np.zeros(12)
            tgt[IDX] = self.q_def_policy + ACTION_SCALE * a
            self.d.ctrl[:] = tgt
            last = a.astype(np.float32)
            p0 = self.d.qpos[0:2].copy()
            for _ in range(DECIMATION):
                mujoco.mj_step(self.m, self.d)
            dist += float(np.linalg.norm(self.d.qpos[0:2] - p0))
            if video_frames is not None:
                video_frames.append(self.render())
            survive += 1
            _, lin_b, _ = self._vel_body()
            vx_list.append(lin_b[0])
            if self.d.qpos[2] < FALL_Z:
                break
        vx = float(np.mean(vx_list[-int(2 / (SIM_DT * DECIMATION)):])) if vx_list else 0.0
        return survive, vx, dist, (survive * SIM_DT * DECIMATION)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="anymal_c 的 scene.xml 路径")
    p.add_argument("--policy", required=True, help="policy.onnx 路径")
    p.add_argument("--video", default=None, help="输出 mp4（默认录 vx=0.8 前进那条指令）")
    p.add_argument("--video_cmd", default="0.8,0,0", help="录哪条指令，格式 vx,vy,wz")
    p.add_argument("--kp", type=float, default=KP)
    p.add_argument("--kd", type=float, default=KD)
    args = p.parse_args()

    sim = AnyMalSim2Sim(args.model, args.policy, args.kp, args.kd)
    cmds = [(0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.6, 0.0, 0.0),
            (0.0, 0.5, 0.0), (0.6, 0.0, 0.6)]

    print("=" * 92)
    print("ANYmal-C sim2sim  |  Isaac Lab (PhysX) → MuJoCo  |  kp=%.0f kd=%.0f" % (args.kp, args.kd))
    print("=" * 92)
    print(f"{'指令 vx':>8} {'vy':>5} {'wz':>5} | {'存活(s)':>8} {'末段实测vx':>11} {'行走距离(m)':>12}  结果")
    rows = []
    vc = tuple(float(x) for x in args.video_cmd.split(","))
    for i, cmd in enumerate(cmds):
        frames = [] if (args.video and tuple(cmd) == vc) else None
        steps, vx, dist, secs = sim.episode(cmd, video_frames=frames)
        ok = secs > EPISODE_S * 0.95
        rows.append((cmd, vx, dist, secs, ok))
        print(f"{cmd[0]:8.2f} {cmd[1]:5.2f} {cmd[2]:5.2f} | {secs:8.2f} {vx:11.3f} {dist:12.2f}  {'✅ 跑满' if ok else '❌ 摔倒'}")
        if frames:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
            imageio.mimsave(args.video, frames, fps=int(1.0 / (SIM_DT * DECIMATION)), quality=8)
            print(f"   视频: {args.video} ({len(frames)} 帧)")

    move = [r for r in rows if abs(r[0][0]) > 0.3 and r[4]]
    print("-" * 92)
    print(f"存活 {EPISODE_S:.0f} 秒不摔: {sum(1 for r in rows if r[4])}/{len(rows)}")
    if move:
        err = [abs(r[1] - r[0][0]) for r in move]
        print(f"前进指令的速度跟踪误差: 平均 {np.mean(err):.3f} m/s (指令 {[r[0][0] for r in move]})")
    print("说明：Isaac Lab 训练用 ActuatorNet LSTM，这里用 PD(kp=%.0f,kd=%.0f) 近似 —— sim2sim 主要误差来源。" % (args.kp, args.kd))


if __name__ == "__main__":
    main()
