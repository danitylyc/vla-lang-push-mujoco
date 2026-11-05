"""Tutorial 05 - 渲染 + 完整一集 oracle：把环境用起来

跑法（仓库根目录）：
    python tutorial/05_render_and_oracle.py

输出：tutorial/out/episode.png（单帧俯视）与终端里的整集距离曲线。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import imageio.v2 as imageio
import mj_env

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)

env = mj_env.LanguagePushEnv(img=256, seed=5)     # 渲染用 256 看清楚
oracle = mj_env.make_oracle()
region = "top-right"

rgb, instr, obs, info = env.reset(region)
print(f"region={region}  区域中心={np.round(mj_env.REGIONS[region][:2], 3)}")
print(f"cube 起点={np.round(obs['achieved_goal'][:2], 3)}")
print(f"指令={mj_env.tokenize_instr_free(region) if hasattr(mj_env, 'tokenize_instr_free') else 'put the cube in the top-right region'}")

# ---- 渲染一帧：Renderer 三件套 ----
renderer = mujoco.Renderer(env._model, 256, 256)
renderer.update_scene(env._data, camera="cam")   # 选 XML 里的 <camera name="cam">
frame = renderer.render()
imageio.imwrite(os.path.join(OUT_DIR, "episode.png"), frame)
print(f"\n已保存 {os.path.join(OUT_DIR, 'episode.png')}  (shape={frame.shape})")

# ---- 整集闭环：oracle 逐步推，打印距离衰减 ----
print("\n=== oracle 整集（最多 60 步）===")
for step in range(60):
    act = oracle(obs)
    rgb2, _, done, info = env.step(act)
    obs = info["obs"]
    d = float(np.linalg.norm(info["obs"]["achieved_goal"][:2] - mj_env.REGIONS[region][:2]))
    if step % 6 == 0 or done:
        print(f"step {step:2d}  dist={d:.3f}  success={info['success']}")
    if done:
        break

# 最后一帧（成功时的位置）
renderer.update_scene(env._data, camera="cam")
imageio.imwrite(os.path.join(OUT_DIR, "episode_final.png"), renderer.render())
print(f"\n最终 success={info['success']}  剩 dist={d:.3f}  已保存 episode_final.png")
renderer.close()
env.close()

print("\n要点：进训练循环时把 env._renderer=None 或直接用 img=64 的 Renderer，")
print("      256×256 只是为了肉眼看得清，训练用会拖慢 4 倍以上。")
