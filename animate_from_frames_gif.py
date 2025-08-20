import glob, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 与 C 程序中的 DT 一致
dt = 0.001

frames = sorted(
    glob.glob("frames/step_*.txt"),
    key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
)
if not frames:
    raise SystemExit("未找到 frames/step_*.txt")

def step_from_path(p):
    return int(re.search(r"(\d+)", os.path.basename(p)).group(1))

d0 = np.loadtxt(frames[0])
x = d0[:, 0]
y_num = d0[:, 1]
y_ex = d0[:, 2]

y_min = min(y_num.min(), y_ex.min())
y_max = max(y_num.max(), y_ex.max())
for f in frames[1:]:
    d = np.loadtxt(f)
    y_min = min(y_min, d[:, 1].min(), d[:, 2].min())
    y_max = max(y_max, d[:, 1].max(), d[:, 2].max())

fig, ax = plt.subplots(figsize=(7, 4))
line_num, = ax.plot(x, y_num, lw=2, label="numerical")
line_ex,  = ax.plot(x, y_ex,  lw=2, ls="--", label="exact")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend()
title = ax.set_title("")

# 额外保存最开始帧的静态 JPG
n0 = step_from_path(frames[0])
t0 = (n0 + 1) * dt  # 与 C 中 next_time 对齐
title.set_text(f"step = {n0}, t = {t0:.6f}")
fig.savefig("dg1d_first_frame.jpg", dpi=200, bbox_inches="tight")

def update(i):
    f = frames[i]
    d = np.loadtxt(f)
    line_num.set_ydata(d[:, 1])
    line_ex.set_ydata(d[:, 2])
    n = step_from_path(f)
    t = (n + 1) * dt  # 与 C 中 next_time 对齐
    title.set_text(f"step = {n}, t = {t:.6f}")
    return line_num, line_ex, title

ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)

# 直接保存为 GIF（需 pillow）
ani.save("dg1d.gif", writer="pillow", fps=5)


