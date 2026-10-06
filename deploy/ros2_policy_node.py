#!/usr/bin/env python3
"""
ROS2 策略节点：把训练好的策略作为实时控制器接入 ROS2。

订阅：
  /joint_states   sensor_msgs/JointState   12 关节角/角速度（顺序 = MJ/SDK 顺序）
  /imu            sensor_msgs/Imu          角速度 + 姿态 → 重力方向
  /odom           nav_msgs/Odometry        机体系线速度（真机来自状态估计器）
  /cmd_vel        geometry_msgs/Twist      vx, vy, wz 速度指令
发布：
  /joint_command  sensor_msgs/JointState   12 关节目标角（50 Hz）

设计要点（与仿真训练一致，也是真机接入时唯一需要替换的部分）：
  * 控制频率 50 Hz（与训练同）
  * 观测 48 维、无归一化；动作 = 默认关节角 + 0.5 × 策略输出
  * 观测延迟可通过 --delay_steps 注入（模拟通信/估计延迟，用于 HIL 式测试）
  * 发布统计：推理延迟、端到端控制周期（用于验证能否守住 20 ms 周期）

运行（WSL + ROS2 Humble）：
  python3 ros2_policy_node.py --policy /path/policy.onnx --cmd 0.8,0,0
"""
import argparse
import statistics
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from anymal_core import ObsBuilder, OnnxPolicy, projected_gravity


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__("anymal_policy")
        self.policy = OnnxPolicy(args.policy)
        self.builder = ObsBuilder()
        self.delay_steps = args.delay_steps
        self.fixed_cmd = np.array([float(v) for v in args.cmd.split(",")]) if args.cmd else None

        self.q = None
        self.qd = None
        self.ang_vel_b = np.zeros(3)
        self.lin_vel_b = np.zeros(3)
        self.grav_b = np.array([0.0, 0.0, -1.0])
        self.cmd = np.zeros(3)
        self.obs_hist = []
        self.infer_ms = []
        self.loop_ms = []
        self._t_last = time.perf_counter()
        self._n = 0

        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)

        period = 1.0 / args.rate
        self.create_timer(period, self.on_control)
        self.get_logger().info(
            f"策略节点已启动：{args.rate:.0f} Hz 控制，观测延迟 {self.delay_steps} 个周期，"
            f"指令 {'固定 ' + args.cmd if self.fixed_cmd is not None else '来自 /cmd_vel'}")

    # ---------------- 订阅回调 ----------------
    def on_js(self, msg: JointState):
        if len(msg.position) == 12:
            self.q = np.asarray(msg.position, dtype=np.float64)
            self.qd = np.asarray(msg.velocity, dtype=np.float64)

    def on_imu(self, msg: Imu):
        o = msg.orientation
        self.ang_vel_b = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.grav_b = projected_gravity([o.w, o.x, o.y, o.z])

    def on_odom(self, msg: Odometry):
        t = msg.twist.twist.linear
        self.lin_vel_b = np.array([t.x, t.y, t.z])

    def on_cmd(self, msg: Twist):
        if self.fixed_cmd is None:
            self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    # ---------------- 控制循环 ----------------
    def on_control(self):
        if self.q is None:
            return
        if self.fixed_cmd is not None:
            self.cmd = self.fixed_cmd

        obs = self.builder.build(self.q, self.qd, self.ang_vel_b, self.lin_vel_b,
                                 self.grav_b, self.cmd)
        self.obs_hist.append(obs)
        if self.delay_steps <= 0:
            used = obs
        elif len(self.obs_hist) > self.delay_steps:
            used = self.obs_hist[-1 - self.delay_steps]
        else:
            used = self.obs_hist[0]

        t0 = time.perf_counter()
        action = self.policy(used)
        infer_ms = (time.perf_counter() - t0) * 1e3
        targets = self.builder.to_joint_targets(action)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i}" for i in range(12)]
        msg.position = [float(x) for x in targets]
        self.pub.publish(msg)

        now = time.perf_counter()
        self.infer_ms.append(infer_ms)
        self.loop_ms.append((now - self._t_last) * 1e3)
        self._t_last = now
        self._n += 1
        if self._n % 250 == 0:                     # 每 5 秒汇报一次
            self.get_logger().info(
                "推理 %.2f ms (p99 %.2f) | 控制周期 %.2f ms (p99 %.2f, 目标 20 ms) | 指令 [%.2f %.2f %.2f]"
                % (statistics.fmean(self.infer_ms[-250:]),
                   np.percentile(self.infer_ms[-250:], 99),
                   statistics.fmean(self.loop_ms[-250:]),
                   np.percentile(self.loop_ms[-250:], 99),
                   *self.cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="policy.onnx")
    ap.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz（训练为 50）")
    ap.add_argument("--delay_steps", type=int, default=0, help="注入的观测延迟（控制周期数）")
    ap.add_argument("--cmd", default=None, help="固定指令 vx,vy,wz；不填则来自 /cmd_vel")
    args, _ = ap.parse_known_args()
    rclpy.init()
    node = PolicyNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
