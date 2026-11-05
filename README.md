# vla-lang-push-mujoco

在 **MuJoCo** 上完整复现语言条件推方块 VLA（PyBullet 版 `vla-lang-push` 的跨平台复刻）。

> 语义层（词表 / 四区域 / oracle 状态机 / 成功判据）与原版逐字对齐，
> 仿真层换成 MuJoCo Franka Emika Panda（`mujoco_menagerie` 资产）+ 自建桌面场景，
> 物理伺服采用每子步 OS-DLS 闭环。同一份 VLA 模型、BC→REINFORCE 训练流水线在两个平台均可复现。

## 任务

桌面上一只立方体（0.08 m）随机出现在四块彩色区域之一，智能体读一句自然语言指令
（如 "put the cube in the top-left region"），用 Franka 的指尖把方块**推到目标区域中心**。

- 观测：64×64 俯视 RGB 图像 + 指令 token 序列（VOCAB 23 词）
- 动作：指尖 2D 平面速度 (dx, dy)（原版 3-D 动作回归中 z 通道恒 0，语义一致）
- 奖励（微调用）：每步 8×(d_prev − d) + 成功 +10
- 成功：方块中心距目标区域中心 < 0.05 m（`max_steps=60`）

## 双平台对比

| 维度 | PyBullet（原版） | MuJoCo（本仓库） |
|---|---|---|
| 机械臂模型 | panda-gym PandaPush-v3（7-DoF） | menagerie franka_emika_panda（7-DoF，nohand） |
| 仿真步进 | Bullet（刚体 + 隐式接触） | MuJoCo 3.17（软接触，100 子步/动作步） |
| 位置伺服 | PyBullet IK + 关节速度 | 每子步 OS-DLS 逆运动学闭环（λ=0.05, Δq≤0.03） |
| 接触掩码 | panda-gym 默认 | cube/tip contype=2·conaffinity=2，桌面 3·3，标记 0 |
| oracle 成功率（100 集） | 96/96 | 69/100（失败全部为 60 步超时，无物理崩溃） |
| oracle 数据（1500 集） | 22313 步（keep-only-success） | 27414 步（1023/1500 集成功 = 68.2%，371 s） |
| BC best val_mse（80 epochs） | 0.00297 @epoch74 | 0.00214 @epoch80 |
| 闭环成功率 seen / unseen / random / oracle | 12/96 · 7/64 · 1/96 · 96/96 | BC 6/72 · 3/48 · 0/72 · 52/72 → **微调后 16/72 · 11/48 · 0/72 · 52/72** |
| 终点距离（微调后） | 0.070 / 0.088 / 0.039 | seen 0.075 · unseen 0.069 · random 0.073 · oracle 0.049 |

> oracle 成功率差异的原因：MuJoCo 侧保留真实关节约束与位形奇异（zeros home 竖直奇异、
> joint4/joint6 ctrlrange 不对称），本仓库通过**抬高基座（+0.60 m）+ 非奇异 home + 指尖接触球**
> 三点修复把成功率从 0–1% 提升到 69%，并把失败全部收敛为超时而非物理失控。

## 目录结构

```
vla-lang-push-mujoco/
├── mj_env.py            # MuJoCo 语言条件推方块环境（语言/oracle/收集/探针）
├── vla_model.py         # VLA 策略（CNN + Transformer + MLP，与原版共用）
├── train_vla.py         # 行为克隆（MSE 回归，90/10 划分）
├── finetune_rl.py       # REINFORCE 闭环微调（分布偏移补偿）
├── eval_vla.py          # 四条件闭环评估（seen/unseen/random/oracle）
├── data/vla_push_train_mj.npz   # 1500 集 oracle 演示（27414 步）
├── checkpoints/best.pt          # BC 最优权重（val_mse 0.00214）
├── checkpoints/finetuned.pt     # 微调后权重（eval_succ 见 README 更新）
└── results/eval_summary.npy     # 四条件评估结果
```

## 复现步骤

```bash
# 1. 环境（已实测）
python -m venv venv
venv\Scripts\pip install mujoco mujoco-menagerie torch numpy gymnasium matplotlib imageio
python -m mujoco_menagerie path franka_emika_panda   # 下载资产到 ~/.mujoco_menagerie/cache

# 2. 生成演示数据（~6 分钟，CPU）
python mj_env.py --mode collect --n 1500 --out data/vla_push_train_mj.npz

# 3. 行为克隆（~20 分钟，CPU）
python train_vla.py --data data/vla_push_train_mj.npz --epochs 80

# 4. REINFORCE 微调（~40 分钟，CPU）
python finetune_rl.py --ckpt checkpoints/best.pt --iters 300

# 5. 四条件闭环评估
python eval_vla.py --ckpt checkpoints/best.pt --oracle
python eval_vla.py --ckpt checkpoints/finetuned.pt --oracle
```

## 已知限制（如实记录）

- oracle 数据源成功率 68.2%（keep-only-success 已剔除失败轨迹，数据干净），
  低于原版演示数据口径；训练质量差异在 README 中如实呈现，不美化。
- 指尖接触用注入球体 `tip_geom`（r=0.022）近似，非夹具真实夹爪。
- 图像渲染 + 每子步 DLS 伺服在 CPU 上约 14 ms/step（含渲染），比 PyBullet 慢（原版无该口径，未对比）。

## 结果更新日志

- [x] 环境稳定：oracle 69/100（四区域各 25 集，失败=超时）
- [x] 数据：1500 集 → 27414 步
- [x] BC：best val_mse 0.00214（80 epochs，CPU 20 分钟）
- [x] 微调：REINFORCE 300 iters，eval 10% → 35%（单次 20 集采样）
- [x] 四条件评估（微调后）：seen 16/72 (22%) · unseen 11/48 (23%) · random 0/72 · oracle 52/72 (72%)
- [x] 双平台对比 + README 完成
