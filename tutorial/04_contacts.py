"""Tutorial 04 - 接触：contype / conaffinity 位掩码 + 实跑推方块

跑法（仓库根目录）：
    python tutorial/04_contacts.py

本项目能跑通的关键之一：让方块只碰桌面和指尖，永远不碰机械臂。
MuJoCo 的规则是位与（AND）：(A.contype & B.conaffinity) 或 (B.contype & A.conaffinity) 非零才检测。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import mj_env

print("=== 本项目位掩码设计 ===")
print("臂 link（menagerie 默认）  : bit1")
print("cube / tip_geom          : contype=2 conaffinity=2  (bit2)")
print("table_top                : contype=3 conaffinity=3  (bit1|bit2)")
print("区域标记 mkr_*            : contype=0 conaffinity=0  (完全不参与碰撞)")
print("=> 臂(bit1) vs cube(bit2) 不相交，方块不会被手臂撞飞")

env = mj_env.LanguagePushEnv(img=64, seed=3)
env._renderer = None
model, data = env._model, env._data

grey = mj_env.make_oracle()
rgb, instr, obs, info = env.reset("top-left")

print("\n=== 跑 12 步 oracle，记录每步接触 ===")
for step in range(12):
    act = grey(obs)
    rgb, _, done, info = env.step(act)
    obs = info["obs"]
    ee = np.round(data.site_xpos[env._site_id], 3)
    cube = np.round(data.xpos[env._cube_id], 3)     # 世界坐标看 xpos，不看 qpos
    names = []
    for c in range(data.ncon):
        ct = data.contact[c]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ct.geom1) or "?"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ct.geom2) or "?"
        names.append(f"{g1}<->{g2}")
    print(f"step {step:2d}  EE={ee}  cube={cube}  ncon={data.ncon}  {'; '.join(names) or '-'}  "
          f"dist={info.get('dist_goal', float('nan')):.3f}" if "dist_goal" in info else
          f"step {step:2d}  EE={ee}  cube={cube}  ncon={data.ncon}  {'; '.join(names) or '-'}")

print("\n=== 接触力从哪读 ===")
print("data.contact[i].frame（接触坐标系）  cfrc_ext / cfrc_local 里的法向力")
print("本项目没有用接触力做控制，只靠'指尖球是否压下方块'这个几何关系推动")

print("\n=== 陷阱复述 ===")
print("cube 是 freejoint：世界位置 = data.xpos[body_id]，qpos[nq-7:] 是 (x,y,z,qw,qx,qy,qz)")
print("camera 里 targetbody 用的是 body id；site 是另一个对象类型（mjOBJ_SITE）")
env.close()
