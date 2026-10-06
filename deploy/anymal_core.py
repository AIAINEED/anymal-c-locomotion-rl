"""
ANYmal-C 部署核心（与 ROS2 / 仿真器无关）。

这里集中了所有经过验证的约定，ROS2 节点、MuJoCo 闭环、真实机器人接口都复用这一份：
  * 关节顺序映射（Isaac Lab 的顺序 ≠ MuJoCo 的顺序）
  * 48 维观测拼装（无缩放、无归一化）
  * 动作 → 关节目标角（默认角 + 0.5 × 策略输出）
  * 执行器 PD 参数（近似 Isaac Lab 的 ActuatorNet LSTM）

真机接入时只需要替换"状态来源"（IMU / 状态估计器）和"指令出口"（力矩接口），
本文件的约定不变。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 常量
MJ_ORDER = ["LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]

POLICY_ORDER = ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
                "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
                "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]

# 策略第 i 维 ↔ MuJoCo（= 真机 SDK 常见的按腿分组顺序）第 IDX[i] 个关节
IDX = [MJ_ORDER.index(j) for j in POLICY_ORDER]

ACTION_SCALE = 0.5          # Isaac Lab: JointPositionActionCfg(scale=0.5, use_default_offset=True)
KP, KD = 120.0, 5.0         # 近似 ActuatorNet LSTM（实测跟踪最好；固件默认 100/1）
EFFORT_LIMIT = 80.0
OBS_DIM, ACT_DIM = 48, 12


def default_joint_pos_mj() -> np.ndarray:
    """Isaac Lab ANYMAL_C_CFG 的初始关节角，按 MuJoCo 顺序排列。"""
    q = np.zeros(12)
    for n in POLICY_ORDER:
        front = n.startswith("LF") or n.startswith("RF")
        q[MJ_ORDER.index(n)] = (0.0 if n.endswith("HAA")
                               else (0.4 if front else -0.4) if n.endswith("HFE")
                               else (-0.8 if front else 0.8))
    return q


DEFAULT_Q_MJ = default_joint_pos_mj()

# 观测：policy 顺序下每个关节的默认角（joint_pos_rel 用）
DEFAULT_Q_POLICY = DEFAULT_Q_MJ[IDX]


class ObsBuilder:
    """把机器人状态拼成策略需要的 48 维观测。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)

    def build(self, q_mj, qd_mj, ang_vel_b, lin_vel_b, projected_gravity_b, cmd) -> np.ndarray:
        """
        q_mj / qd_mj          : 关节角/角速度（MuJoCo 或真机 SDK 顺序）
        ang_vel_b / lin_vel_b : 机体坐标系角速度/线速度（rad/s, m/s）
        projected_gravity_b   : 重力方向在机体系下的表示（IMU 给出）
        cmd                   : [vx, vy, wz]（机体系）
        """
        q_pol = np.asarray(q_mj, dtype=np.float64)[IDX]
        qd_pol = np.asarray(qd_mj, dtype=np.float64)[IDX]
        obs = np.concatenate([
            np.asarray(lin_vel_b, dtype=np.float64),
            np.asarray(ang_vel_b, dtype=np.float64),
            np.asarray(projected_gravity_b, dtype=np.float64),
            np.asarray(cmd, dtype=np.float64),
            q_pol - DEFAULT_Q_POLICY,
            qd_pol,
            self.last_action.astype(np.float64),
        ])
        assert obs.shape == (OBS_DIM,), obs.shape
        return obs.astype(np.float32)

    def to_joint_targets(self, action) -> np.ndarray:
        """策略输出 → 关节目标角（MuJoCo / 真机顺序）。同时记录 last_action。"""
        action = np.asarray(action, dtype=np.float32)
        self.last_action = action.copy()
        tgt = DEFAULT_Q_MJ.copy()
        tgt[IDX] += ACTION_SCALE * action
        return tgt


class OnnxPolicy:
    """ONNX 策略推理（onnxruntime，CPU；不依赖 Isaac Sim）。

    实测：batch=1 单次推理 p99 < 1 ms（i5-11400），远快于 20 ms 控制周期。
    """

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.output_name], {self.input_name: obs[None].astype(np.float32)})[0]
        return out[0].astype(np.float32)


def quat_to_rotmat(quat_wxyz) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵。"""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def projected_gravity(quat_wxyz) -> np.ndarray:
    """重力方向在机体坐标系下的表示（IMU 的典型输出）。"""
    return quat_to_rotmat(quat_wxyz).T @ np.array([0.0, 0.0, -1.0])
