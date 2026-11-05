# MuJoCo 教学：用本项目当教材

五节课，全部基于本仓库真实代码，每个脚本都能直接跑、有真实输出。

```powershell
$py = "C:\Users\12698\venvs\mujoco\Scripts\python.exe"
cd C:\Users\12698\Doubao\chats\2026-11-02\new-chat\Portfolio\vla-lang-push-mujoco

& $py tutorial\01_minimal_model.py
& $py tutorial\02_anatomy.py
& $py tutorial\03_kinematics.py
& $py tutorial\04_contacts.py
& $py tutorial\05_render_and_oracle.py
```

| 课 | 文件 | 学什么 | 对应项目代码 |
|---|---|---|---|
| 01 | `01_minimal_model.py` | `MjModel` / `MjData` / `mj_step` 三件套；`nq` vs `nv` 为何不同；freejoint 布局 | 一切的基础 |
| 02 | `02_anatomy.py` | 编译后的 id 映射：body 树、7 个 joint 的 qpos/dof 起始位、actuator kp 与 ctrlrange、geom 位掩码、`<pair>` | `mj_env._build_model` |
| 03 | `03_kinematics.py` | 正解 `site_xpos`、`mj_jacSite` 雅可比、阻尼最小二乘（DLS）IK 收敛 | `mj_env.LanguagePushEnv._ik_step` |
| 04 | `04_contacts.py` | `contype`/`conaffinity` 位与规则、实跑读 `data.ncon` / `data.contact`、freejoint 世界坐标陷阱 | `_SCENE_EXTRA` 的位掩码设计 |
| 05 | `05_render_and_oracle.py` | `mujoco.Renderer` 三件套、相机 `targetbody`、oracle 整集闭环 + 出图 | `LanguagePushEnv.render` / `collect` |

## 每节课后的问题（自己动手，别只看）

- **01**：把 `<option timestep>` 改成 0.01，球落到底的时间变了吗？`ncon` 什么时候变 1？
- **02**：`joint4` 的限位是 [-3.072, -0.07]，为什么不是对称的 ±2.897？如果 `ctrl=0` 会发生什么？
- **03**：把 `lam` 从 0.05 改成 0（纯伪逆），在 home 附近还能收敛吗？把 target 改到臂可达边界外会怎样？
- **04**：把 `cube_geom` 的 `contype` 改成 0（其余不变），碰撞对还在吗？（本项目实测：连显式 pair 一起失效）
- **05**：把 `img` 从 256 改成 64，跑同样一集，看耗时差几倍；这解释了训练时为什么用 64。

## 三条最容易踩的坑（都在这项目里被踩过）

1. **freejoint 的世界位置不是 `qpos`**。`cube` 是 freejoint：`qpos[nq-7:]` 是 `(x,y,z,qw,qx,qy,qz)`，世界坐标读 `data.xpos[body_id]`。
2. **雅可比要切列**。`mj_jacSite` 返回 `(3, nv)`，`nv=13` 含方块的 6 个自由度；控制机械臂只能用 `J[:, :7]`。
3. **改了 `qpos` 必须 `mj_forward`**。否则 `site_xpos` / 雅可比还是旧值；`mj_step` 才会推进时间。
