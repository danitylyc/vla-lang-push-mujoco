"""Tutorial 01 - MuJoCo 最小闭环：model / data / step

跑法：
    python tutorial/01_minimal_model.py

你会看到 MuJoCo 的三件套：MjModel（编译后的静态模型）、MjData（全部动态状态）、mj_step。
"""
import mujoco
import numpy as np

XML = """
<mujoco model="ball_drop">
  <option timestep="0.005"/>
  <worldbody>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0.3 0.3 1"/>
    </body>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.6 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("=== 模型规模（编译期常量）===")
print(f"nq   = {model.nq}   # 广义坐标个数（freejoint 7 个）")
print(f"nv   = {model.nv}   # 广义速度个数（freejoint 6 个）")
print(f"nu   = {model.nu}   # actuator 个数（这里 0，纯被动落体）")
print(f"nbody= {model.nbody}  ngeom = {model.ngeom}  njnt = {model.njnt}")
print(f"timestep = {model.opt.timestep}")

print("\n=== qpos 布局（世界位置 + 四元数）===")
print("qpos 初值 =", np.round(data.qpos, 3), " (x y z qw qx qy qz)")

print("\n=== 推进 300 步，每 30 步打印球心高度 ===")
for i in range(300):
    mujoco.mj_step(model, data)
    if i % 30 == 29:
        # freejoint 刚体的世界位置用 xpos，不是 qpos（qpos 只是刚体局部根位置）
        print(f"t={data.time:5.2f}s  z(xpos)={data.xpos[model.body('ball').id][2]:.3f}  "
              f"qvel_z={data.qvel[2]:+.3f}  ncon={data.ncon}")

print("\n要点：model 是常量，data 是每步被 mj_step 改写的全部状态；")
print("      freejoint 物体的世界坐标看 data.xpos[body_id]，看 qpos 会被四元数/局部偏移坑。")
