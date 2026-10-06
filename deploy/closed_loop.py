"""
部署栈闭环评估（不依赖 ROS2，纯 Python）：策略 ↔ MuJoCo 被控对象。
用于回答部署级问题：
  * 50 Hz 控制周期能不能守住（单次推理延迟 / 整环耗时 p99）
  * 观测延迟（0/1/2/3 个控制周期 = 0/20/40/60 ms）下策略还站得住吗
  * 机体速度估计有噪声时表现如何

用法：
  python closed_loop.py --model <...>/scene.xml --policy policy.onnx \
         --episodes 20 --delays 0,1,2,3 --vel_noise 0.0 --csv deploy_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from anymal_core import ObsBuilder, OnnxPolicy
from anymal_plant import AnyMalPlant

EPISODE_S = 20.0
FALL_Z = 0.35


def run_episode(plant, policy, builder, cmd, delay_steps=0, vel_noise=0.0, rng=None,
                max_steps=None, video_frames=None):
    plant.reset()
    builder.reset()
    n = int(EPISODE_S / plant.control_dt)
    if max_steps:
        n = min(n, max_steps)

    obs_hist = []
    infer_t = []
    loop_t = []
    vx_err = []
    survived = 0

    for _ in range(n):
        st = plant.state()
        lin = st["lin_vel_b"]
        if vel_noise > 0 and rng is not None:
            lin = lin + rng.normal(0.0, vel_noise, size=3)      # 速度估计噪声（真机上的典型误差源）
        obs = builder.build(st["q"], st["qd"], st["ang_vel_b"], lin,
                            st["projected_gravity_b"], cmd)
        obs_hist.append(obs)

        # 观测延迟：delay_steps=0 用当前帧；>0 用 delay_steps 个控制周期之前的帧
        # （历史不足时退回最早一帧 —— 对应"上电初期"）
        if delay_steps <= 0:
            used = obs
        elif len(obs_hist) > delay_steps:
            used = obs_hist[-1 - delay_steps]
        else:
            used = obs_hist[0]

        t0 = time.perf_counter()
        action = policy(used)
        infer_t.append((time.perf_counter() - t0) * 1e3)         # ms

        plant.set_joint_targets(builder.to_joint_targets(action))
        plant.advance_control_step()

        st2 = plant.state()
        vx_err.append(abs(st2["lin_vel_b"][0] - cmd[0]))
        survived += 1
        loop_t.append((time.perf_counter() - t0) * 1e3)          # 整环耗时（推理+物理）ms

        if video_frames is not None:
            video_frames.append(plant.render())
        if st2["height"] < FALL_Z:
            break

    return {
        "survived": survived,
        "survival_s": survived * plant.control_dt,
        "fell": survived < n,
        "track_err": float(np.mean(vx_err)) if vx_err else 0.0,
        "infer_mean_ms": float(np.mean(infer_t)),
        "infer_p99_ms": float(np.percentile(infer_t, 99)),
        "loop_mean_ms": float(np.mean(loop_t)),
        "loop_p99_ms": float(np.percentile(loop_t, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--delays", default="0,1,2,3", help="观测延迟（控制周期数，1 = 20 ms）")
    ap.add_argument("--vel_noise", type=float, default=0.0, help="机体线速度估计噪声标准差 (m/s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="deploy_eval.csv")
    ap.add_argument("--video", default=None, help="录制第一个 episode（0 延迟）")
    args = ap.parse_args()

    plant = AnyMalPlant(args.model)
    policy = OnnxPolicy(args.policy)
    builder = ObsBuilder()
    rng = np.random.default_rng(args.seed)

    print("=" * 96)
    print("部署栈闭环评估 | 控制周期 %.0f ms (%.0f Hz) | 每个配置 %d 个 20 秒 episode"
          % (plant.control_dt * 1e3, 1.0 / plant.control_dt, args.episodes))
    print("=" * 96)
    print(f"{'延迟':>8} {'摔倒率':>8} {'平均存活(s)':>11} {'跟踪误差':>9} "
          f"{'推理p99(ms)':>12} {'整环p99(ms)':>12} {'实时?':>6}")

    rows = []
    for di, d in enumerate(args.delays.split(",")):
        delay = int(d)
        res = []
        for ep in range(args.episodes):
            cmd = rng.uniform(-1.0, 1.0, size=3)          # 与训练分布一致：vx,vy,wz ∈ [-1,1]
            frames = [] if (args.video and delay == 0 and ep == 0) else None
            res.append(run_episode(plant, policy, builder, cmd, delay_steps=delay,
                                   vel_noise=args.vel_noise, rng=rng, video_frames=frames))
            if frames:
                import imageio
                os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
                imageio.mimsave(args.video, frames, fps=int(1.0 / plant.control_dt), quality=8)
                print(f"   视频已保存: {args.video} ({len(frames)} 帧)")

        fall_rate = np.mean([r["fell"] for r in res])
        surv = np.mean([r["survival_s"] for r in res])
        err = np.mean([r["track_err"] for r in res])
        ip99 = np.max([r["infer_p99_ms"] for r in res])
        lp99 = np.max([r["loop_p99_ms"] for r in res])
        realtime = "是" if lp99 < plant.control_dt * 1e3 else "否"
        rows.append([delay, round(delay * plant.control_dt * 1e3, 1), args.episodes,
                     int(round(fall_rate * args.episodes)), round(fall_rate, 4),
                     round(surv, 2), round(err, 4), round(ip99, 3), round(lp99, 3), args.vel_noise])
        print(f"{delay:>5}周期 {fall_rate:8.2%} {surv:11.2f} {err:9.4f} "
              f"{ip99:12.3f} {lp99:12.3f} {realtime:>6}")

    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["delay_steps", "delay_ms", "episodes", "falls", "fall_rate",
                        "mean_survival_s", "mean_track_err", "infer_p99_ms", "loop_p99_ms",
                        "vel_noise_std"])
        w.writerows(rows)
    print("-" * 96)
    print(f"结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
