"""Tutorial 03 - Franka 运动学：正解 site_xpos + 雅可比 + DLS 逆解

跑法（仓库根目录）：
    python tutorial/03_kinematics.py

这是本项目"动作 -> 关节"的核心：策略只输出指尖的 2D 速度，环境用雅可比把它
变成 7 个关节角。看懂这个脚本，你就看懂了 mj_env.step 里的 OS-DLS 伺服。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import mj_env

model = mj_env._build_model()
data = mujoco.MjData(model)
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

# ---------- 1. 正运动学：给关节角，问末端在哪 ----------
data.qpos[:7] = mj_env._HOME_QPOS
mujoco.mj_forward(model, data)        # 重算所有派生量（xpos / site_xpos / 雅可比都要它）
print("home qpos =", np.round(mj_env._HOME_QPOS, 3))
print("EE        =", np.round(data.site_xpos[site], 4))

# 动一下 joint0（绕 z 轴转 30 度），看末端怎么走
data.qpos[:7] = mj_env._HOME_QPOS
data.qpos[0] += np.deg2rad(30)
mujoco.mj_forward(model, data)
print("\njoint0 +30° -> EE =", np.round(data.site_xpos[site], 4))

# ---------- 2. 雅可比：末端速度 = J @ qdot ----------
data.qpos[:7] = mj_env._HOME_QPOS
mujoco.mj_forward(model, data)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site)
J = jacp[:, :7]      # 前 7 列 = 机械臂 7 个关节；后面列属于 cube 的 freejoint！
print("\n雅可比 J 形状 =", J.shape, " (nv=13: 7 臂 + 6 freejoint)")
print("只用臂的 7 列，即 J[:, :7] —— 忘了切会把方块的自由度也算进去")

# ---------- 3. 一步 DLS 逆解：把末端推向目标 ----------
target = np.array([0.55, 0.30, mj_env.TABLE_TOP + 0.05])
print(f"\n目标点 = {np.round(target, 3)}")
data.qpos[:7] = mj_env._HOME_QPOS
mujoco.mj_forward(model, data)

lam = 0.05
for it in range(40):
    ee = data.site_xpos[site]
    err = target - ee
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    J = jacp[:, :7]
    A = J @ J.T + lam * np.eye(3)          # 阻尼最小二乘，避免奇异处爆炸
    dq = J.T @ np.linalg.solve(A, err)
    dq = np.clip(dq, -0.05, 0.05)          # 每步限幅，和项目里 dq_max 同一个作用
    data.qpos[:7] += dq
    mujoco.mj_forward(model, data)
    if it % 8 == 0 or it == 39:
        print(f"iter {it:2d}  |err|={np.linalg.norm(err):.4f} m  EE={np.round(data.site_xpos[site], 3)}")

print("\n要点：mj_forward 是'只算不推进'；mj_step 是'算 + 按时间积分'。")
print("      IK 迭代里必须每轮 mj_forward 重算雅可比（雅可比随关节角变化）。")
