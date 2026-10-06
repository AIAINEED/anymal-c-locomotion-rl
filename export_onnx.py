"""
把 skrl 训练出的 checkpoint 导出为 ONNX，并做数值一致性验证。

用法（Windows，Isaac Sim 自带 Python）：
  E:\\IsaacLab\\_isaac_sim\\kit\\python\\python.exe export_onnx.py

实测结果：ONNX vs PyTorch 最大误差 9.537e-07
"""
import os

import numpy as np
import torch
import torch.nn as nn

CKPT = r"E:\IsaacLab\logs\skrl\anymal_c_flat\2026-10-03_01-51-51_ppo_torch\checkpoints\best_agent.pt"
OUT = r"E:\IsaacLab\deploy"
OBS, ACT, H = 48, 12, 128


class Actor(nn.Module):
    """skrl 的 policy 网络：3 层 MLP（128-128-128，ELU）→ 动作均值。无观测归一化。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS, H), nn.ELU(),
            nn.Linear(H, H), nn.ELU(),
            nn.Linear(H, H), nn.ELU(),
            nn.Linear(H, ACT),
        )

    def forward(self, x):
        return self.net(x)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) 加载 checkpoint（skrl 保存的是 agent 的 state_dict）
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["policy"]

    # 2) 键名映射：skrl 的 net_container.* + policy_layer.* → 我们的 nn.Sequential
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net_container."):
            new_sd[k.replace("net_container.", "net.")] = v
        elif k.startswith("policy_layer."):
            new_sd["net.6." + k.split(".")[1]] = v

    model = Actor()
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    torch.save(model.state_dict(), os.path.join(OUT, "actor.pt"))

    # 3) 导出 ONNX（动态 batch，便于批量部署）
    onnx_path = os.path.join(OUT, "policy.onnx")
    torch.onnx.export(
        model, torch.zeros(1, OBS), onnx_path,
        input_names=["obs"], output_names=["action"], opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    # 4) 数值一致性验证
    import onnxruntime as ort
    x = torch.randn(16, OBS)
    with torch.no_grad():
        y = model(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y2 = sess.run(None, {"obs": x.numpy()})[0]

    lines = [
        "EXPORT OK -> " + onnx_path,
        "ONNX vs PyTorch max err: %.3e" % float(np.abs(y - y2).max()),
    ]
    # 5) 零指令 sanity check（重力项 = [0,0,-1]）
    z = np.zeros((1, OBS), np.float32)
    z[0, 6:9] = [0.0, 0.0, -1.0]
    with torch.no_grad():
        a = model(torch.from_numpy(z)).numpy()[0]
    lines.append("zero-cmd action: " + str(np.round(a, 3).tolist()))

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "export_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
