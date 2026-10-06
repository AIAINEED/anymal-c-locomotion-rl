"""
推力鲁棒性评估：同一策略在不同外部推力强度下的存活率与速度跟踪误差。

注意：**必须用 torch 加载权重，不能用 onnxruntime** —— onnxruntime 的原生扩展在 Kit
进程里会崩（Kit 自带的旧 CRT 与新编译扩展冲突，与 issaclab 的 tensordict 是同一个问题）。

用法（在 E:\\IsaacLab 目录下，用 Isaac Lab 的启动器）：
  E:\\IsaacLab\\isaaclab.bat -p robustness_eval.py --headless ^
      --actor E:\\IsaacLab\\deploy\\actor.pt --intensity 0.0 --num_envs 64 --episodes 3 --csv robustness.csv

  # 或直接读 skrl checkpoint：
  --checkpoint E:\\IsaacLab\\logs\\skrl\\anymal_c_flat\\<run>\\checkpoints\\best_agent.pt

intensity = 0 表示不施加推力（基线）；>0 时每 2–4 秒施加一次 ±intensity m/s 的速度扰动。
"""
import argparse
import csv
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
p.add_argument("--task", default="Isaac-Velocity-Flat-Anymal-C-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--actor", default=None, help="actor.pt（export_onnx.py 导出）")
p.add_argument("--checkpoint", default=None, help="skrl best_agent.pt（与 --actor 二选一）")
p.add_argument("--intensity", type=float, default=0.0, help="推力强度 m/s，0 = 不推")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--csv", default="robustness.csv")
args = p.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


class Actor(nn.Module):
    """与训练时一致的策略网络：3 层 MLP（128-128-128，ELU）。"""

    def __init__(self, obs=48, act=12, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, act))

    def forward(self, x):
        return self.net(x)


def load_policy(a, device):
    model = Actor()
    if a.actor:
        sd = torch.load(a.actor, map_location="cpu")
    else:
        sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["policy"]
        # skrl 的键名 → nn.Sequential 的键名
        sd = {("net." + k.split("net_container.")[1]) if k.startswith("net_container.")
              else ("net.6." + k.split(".")[1]) if k.startswith("policy_layer.") else k: v
              for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith(("value_layer", "log_std"))}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device)


cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.observations.policy.enable_corruption = False
if args.intensity <= 0.0:
    cfg.events.push_robot = None
else:
    cfg.events.push_robot.interval_range_s = (2.0, 4.0)
    cfg.events.push_robot.params["velocity_range"] = {
        "x": (-args.intensity, args.intensity), "y": (-args.intensity, args.intensity)}

env = gym.make(args.task, cfg=cfg).unwrapped
policy = load_policy(args, env.device)
step_dt = float(getattr(env, "step_dt", cfg.decimation * cfg.sim.dt))
max_steps = int(20.0 / step_dt) * args.episodes

res = env.reset()
obs = res[0] if isinstance(res, tuple) else res
falls = timeouts = 0
err_sum, err_n = 0.0, 0
for _ in range(max_steps):
    with torch.no_grad():
        act = policy(obs["policy"])
    out = env.step(act)
    if len(out) == 5:
        obs, rew, term, trunc, info = out
    else:
        obs, rew, done, info = out
        term, trunc = done, torch.zeros_like(done)
    falls += int(term.sum().item())
    timeouts += int(trunc.sum().item())
    v = env.scene["robot"].data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command("base_velocity")[:, 0]
    err_sum += float((v - cmd).abs().sum().item())
    err_n += int(v.numel())

total = falls + timeouts
fall_rate = falls / total if total else 0.0
mean_err = err_sum / err_n if err_n else 0.0
print("intensity=%.2f  episodes=%d  falls=%d  fall_rate=%.3f  track_err=%.4f"
      % (args.intensity, total, falls, fall_rate, mean_err))

new = not os.path.exists(args.csv)
with open(args.csv, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["push_intensity_mps", "episodes", "falls", "timeouts", "fall_rate", "mean_abs_vx_err"])
    w.writerow([args.intensity, total, falls, timeouts, round(fall_rate, 4), round(mean_err, 4)])

os._exit(0)  # Kit 的 sim_app.close() 有时不返回，直接退出（CSV 已写入）
