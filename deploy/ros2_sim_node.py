#!/usr/bin/env python3
"""
ROS2 被控对象节点：用 MuJoCo 扮演"真机"。

发布（模拟真机传感器）：
  /joint_states   sensor_msgs/JointState   12 个关节角/角速度（顺序 = MJ/SDK 顺序）
  /imu            sensor_msgs/Imu          机体角速度 + 姿态（策略从中得到重力方向）
  /odom           nav_msgs/Odometry        机体坐标系线速度（真机上来自状态估计器）
订阅（来自控制器）：
  /joint_command  sensor_msgs/JointState   12 个关节目标角

物理 200 Hz（dt=0.005），每 4 步发布一次状态（= 50 Hz 控制周期，与训练一致）。

运行（WSL + ROS2 Humble）：
  python3 ros2_sim_node.py --model /path/anymal_c/scene.xml
"""
import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from anymal_core import DEFAULT_Q_MJ, MJ_ORDER
from anymal_plant import AnyMalPlant


class SimNode(Node):
    def __init__(self, args):
        super().__init__("anymal_sim")
        self.plant = AnyMalPlant(args.model)
        self.decimation = AnyMalPlant.DECIMATION
        self.tick = 0
        self.targets = DEFAULT_Q_MJ.copy()
        self.last_cmd_time = self.get_clock().now()

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "/imu", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(JointState, "/joint_command", self.on_command, 10)

        self.create_timer(AnyMalPlant.SIM_DT, self.on_physics)   # 200 Hz
        self.get_logger().info(
            f"MuJoCo ANYmal-C 已启动：物理 200 Hz / 状态发布 50 Hz，关节顺序 = {MJ_ORDER[0]} ...")

    def on_command(self, msg: JointState):
        if len(msg.position) == 12:
            self.targets = np.asarray(msg.position, dtype=np.float64)
            self.last_cmd_time = self.get_clock().now()

    def on_physics(self):
        self.plant.set_joint_targets(self.targets)
        self.plant.advance_control_step()
        self.tick += 1
        if self.tick % self.decimation == 0:      # 每 4 个物理步 = 50 Hz
            self.publish_state()

    def publish_state(self):
        st = self.plant.state()
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = list(MJ_ORDER)
        js.position = [float(x) for x in st["q"]]
        js.velocity = [float(x) for x in st["qd"]]
        self.pub_js.publish(js)

        w, x, y, z = [float(v) for v in st["quat_wxyz"]]
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base"
        imu.orientation = Quaternion(x=x, y=y, z=z, w=w)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = \
            [float(v) for v in st["ang_vel_b"]]
        imu.linear_acceleration.z = 9.81          # 站立时近似（真机由 IMU 给出）
        self.pub_imu.publish(imu)

        # base_lin_vel 真机上由状态估计器给出，这里直接用 MuJoCo 的机体系线速度
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        odom.pose.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = \
            [float(v) for v in st["lin_vel_b"]]
        self.pub_odom.publish(odom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="anymal_c scene.xml")
    args, _ = ap.parse_known_args()          # 忽略 ROS2 自己的参数
    rclpy.init()
    node = SimNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
