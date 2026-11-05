"""Tutorial 02 - 解剖本项目场景：body 树 / joint / geom / actuator / 位掩码

跑法（在仓库根目录）：
    python tutorial/02_anatomy.py

这个脚本直接加载 mj_env 的真实场景（Franka + 桌面 + 方块 + 四个标记 + 相机），
把 XML 编译后的 id 映射全部打印出来 —— 以后写任何 MuJoCo 代码，你都是靠这些 id 定位。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np
import mj_env

def np_round(v, n=3):
    return np.round(np.asarray(v, dtype=float), n).tolist()

model = mj_env._build_model()
data = mujoco.MjData(model)

print("=== 规模 ===")
print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}  ngeom={model.ngeom}")

print("\n=== body 树（id  parent  name  局部pos）===")
for b in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "(unnamed)"
    parent = model.body_parentid[b]
    pos = np_round(model.body_pos[b])
    print(f"{b:3d} {parent:3d}  {name:22s} {pos}")

print("\n=== joint 布局（name  type  qpos 起始  dof 起始  限位）===")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "(unnamed)"
    jt = model.jnt_type[j]
    qadr, dadr = model.jnt_qposadr[j], model.jnt_dofadr[j]
    limited = "自由" if model.jnt_limited[j] == 0 else f"限位{np_round(model.jnt_range[j], 3)}"
    print(f"{j:3d}  {name:22s} type={jt}  qpos[{qadr}]  dof[{dadr}]  {limited}")

print("\n=== actuator（7 个位置伺服）===")
for a in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "(unnamed)"
    print(f"{a:2d} {name:22s} ctrlrange={np_round(model.actuator_ctrlrange[a], 3)} "
          f"kp={model.actuator_gainprm[a][0]:.0f} dyntype={model.actuator_dyntype[a]}")

print("\n=== 关键 geom（碰撞位掩码）===")
for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if not name:
        continue
    if name in ("cube_geom", "tip_geom", "table_top", "floor") or name.startswith("mkr"):
        print(f"{name:12s} contype={model.geom_contype[g]}  conaffinity={model.geom_conaffinity[g]}")

print("\n=== 显式接触对 ===")
nbp = model.npair
for i in range(nbp):
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom1[i])
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom2[i])
    print(f"pair {i}: {g1} <-> {g2}")

print("\n=== 关键 id（代码里靠这些常量定位，比字符串查找快）===")
for obj, nm in [(mujoco.mjtObj.mjOBJ_SITE, "attachment_site"),
                (mujoco.mjtObj.mjOBJ_BODY, "cube"),
                (mujoco.mjtObj.mjOBJ_BODY, "table"),
                (mujoco.mjtObj.mjOBJ_CAMERA, "cam")]:
    print(f"{nm:18s} id={mujoco.mj_name2id(model, obj, nm)}")

print("\n=== home 姿态下的末端 ===")
data.qpos[:7] = mj_env._HOME_QPOS
mujoco.mj_forward(model, data)
site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
print("EE(site_xpos) =", np_round(data.site_xpos[site]))
print("立方体(xpos)  =", np_round(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")]))
print("桌面顶 z      =", mj_env.TABLE_TOP)
