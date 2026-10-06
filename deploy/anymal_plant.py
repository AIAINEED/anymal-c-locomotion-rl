"""
MuJoCo 被控对象：把 ANYmal-C 当作"真机"来驱动（位置指令进、关节状态出）。

与 anymal_sim2sim.py 的区别：这里只做"被控对象"的角色 —— 接收 12 个关节目标角，
按 50 Hz 控制周期推进物理（内部 200 Hz），输出关节状态与机体状态，
用于验证部署栈（ROS2 节点 / 硬件约束鲁棒性）而不引入 Isaac Sim。
"""
from __future__ import annotations

import numpy as np
import mujoco

from anymal_core import (DEFAULT_Q_MJ, EFFORT_LIMIT, KD, KP, quat_to_rotmat)


class AnyMalPlant:
    SIM_DT = 0.005
    DECIMATION = 4          # 控制周期 = 0.02 s → 50 Hz

    def __init__(self, xml_path: str, base_init_z: float = 0.62):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._v = np.zeros(6)

        for i in range(self.m.nu):
            self.m.actuator_gainprm[i, 0] = KP
            self.m.actuator_biasprm[i, 1] = -KP
            self.m.actuator_biasprm[i, 2] = -KD
            self.m.actuator_forcerange[i] = [-EFFORT_LIMIT, EFFORT_LIMIT]

        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if gid >= 0:
            self.m.geom_friction[gid] = [1.0, 0.005, 0.0001]

        self.base_init_z = base_init_z
        self.reset()

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0.0, 0.0, self.base_init_z]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[7:] = DEFAULT_Q_MJ
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def _vel(self):
        """机体坐标系线/角速度。注意 MuJoCo 的 flg_local=1 返回值有误，必须用 0 再自己旋转。"""
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.base, self._v, 0)
        R = self.d.xmat[self.base].reshape(3, 3)
        return R.T @ self._v[0:3], R.T @ self._v[3:6], R

    def state(self) -> dict:
        ang_b, lin_b, R = self._vel()
        q = self.d.qpos[7:].copy()
        qd = self.d.qvel[6:].copy()
        return {
            "q": q,                                   # MuJoCo 顺序
            "qd": qd,
            "ang_vel_b": ang_b,
            "lin_vel_b": lin_b,
            "projected_gravity_b": R.T @ np.array([0.0, 0.0, -1.0]),
            "height": float(self.d.qpos[2]),
            "quat_wxyz": self.d.qpos[3:7].copy(),
            "base_xy": self.d.qpos[0:2].copy(),
        }

    # ---------------------------------------------------------------- 驱动
    def set_joint_targets(self, targets) -> None:
        self.d.ctrl[:] = np.asarray(targets, dtype=np.float64)

    def advance_control_step(self) -> None:
        """推进一个控制周期（内部 DECIMATION 个物理步）。"""
        for _ in range(self.DECIMATION):
            mujoco.mj_step(self.m, self.d)

    @property
    def control_dt(self) -> float:
        return self.SIM_DT * self.DECIMATION

    def render(self, camera=None):
        if not hasattr(self, "_r"):
            self._r = mujoco.Renderer(self.m, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 3.2, 130.0, -18.0
        cam = camera or self._cam
        cam.lookat[:] = self.d.qpos[0:3]
        self._r.update_scene(self.d, camera=cam)
        return self._r.render().copy()
