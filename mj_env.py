"""VLA-Push-MuJoCo: the vla-lang-push task reproduced on the MuJoCo physics engine.

Same task semantics as the PyBullet original (vla-lang-push/vla_env.py):
  image (64x64 RGB) + natural-language instruction -> 3D end-effector delta action,
  Franka arm pushes a cube on a table into one of four language-specified regions.

This module provides:
  * identical language vocabulary / regions / instruction templates (copied verbatim
    from vla_env.py so the two projects share the same tokenizer and eval protocol)
  * a MuJoCo environment (Franka Emika Panda from mujoco_menagerie, position-servo
    joints, DLS inverse kinematics for the 3-D end-effector delta action)
  * the same analytic oracle policy (approach -> lower -> push) used to generate
    demonstrations
  * dataset collection into the same .npz schema as the original project

Usage:
  python mj_env.py probe            # smoke-test env + render timing
  python mj_env.py verify-oracle    # measure oracle closed-loop success rate
  python mj_env.py collect --n 1200 --out data/vla_push_train.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# model / scene
# ---------------------------------------------------------------------------
# menagerie panda (no hand) model directory; the scene is assembled at runtime by
# injecting a table + cube + target markers + camera into the menagerie XML.
PANDA_NO_HAND_XML = os.path.join(
    os.environ.get("MUJOCO_MENAGERIE_CACHE",
                   r"C:\Users\12698\AppData\Local\mujoco_menagerie\cache\models"
                   r"\franka_emika_panda-386f680ac533760a"),
    "panda_nohand.xml")

# table-top plane geometry (world frame)
TABLE_X, TABLE_Y = 0.0, 0.72     # table centre on the ground plane (on the +y side of the arm)
TABLE_TOP = 0.60                  # table-top surface height
_BASE_Z = 0.60                    # Franka base raised to table-top height: the arm chain
                                  # then always works at z >= TABLE_TOP and never collides
                                  # with the table while entering the workspace
_CUBE_HALF = 0.04                 # cube half-extent
CUBE_Z = TABLE_TOP + _CUBE_HALF   # cube centre height
EE_H = TABLE_TOP + 0.030          # push height (fingertip centre, below cube centre)
FLY_H = TABLE_TOP + 0.100         # approach height (above cube top)
REGION_R = 0.075

# region -> (name, centre_xyz in world frame)
REGIONS = {
    "top-left":     np.array([TABLE_X - REGION_R, TABLE_Y + REGION_R, CUBE_Z]),
    "top-right":    np.array([TABLE_X + REGION_R, TABLE_Y + REGION_R, CUBE_Z]),
    "bottom-left":  np.array([TABLE_X - REGION_R, TABLE_Y - REGION_R, CUBE_Z]),
    "bottom-right": np.array([TABLE_X + REGION_R, TABLE_Y - REGION_R, CUBE_Z]),
}
REGION_NAMES = list(REGIONS.keys())

# instruction templates -> (region key). Multi-word forms per region (verbatim
# from vla_env.py).
_TEMPLATES = {
    "top-left": [
        "push the green cube to the top left target",
        "move the cube to the upper left target",
        "push the block towards the top left",
    ],
    "top-right": [
        "push the green cube to the top right target",
        "move the cube to the upper right target",
        "push the block towards the top right",
    ],
    "bottom-left": [
        "push the green cube to the bottom left target",
        "move the cube to the lower left target",
        "push the block towards the bottom left",
    ],
    "bottom-right": [
        "push the green cube to the bottom right target",
        "move the cube to the lower right target",
        "push the block towards the bottom right",
    ],
}
TEMPLATE_POOL = [(r, t) for r, ts in _TEMPLATES.items() for t in ts]

# fixed small vocabulary (verbatim from vla_env.py)
VOCAB = ["<pad>", "<cls>",
         "push", "move", "put", "the", "green", "cube", "block", "to", "target",
         "in", "at", "top", "bottom", "left", "right", "upper", "lower",
         "towards", "please", "region", "corner"]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}
MAX_INSTR_LEN = 8

# hold-out instruction phrasings, unseen at training time (verbatim from vla_env.py)
EVAL_ONLY_TEMPLATES = {
    "top-left":      ["put the cube in the top left region", "at the upper left corner please"],
    "top-right":     ["put the cube in the top right region", "at the upper right corner please"],
    "bottom-left":   ["put the cube in the bottom left region", "at the lower left corner please"],
    "bottom-right":  ["put the cube in the bottom right region", "at the lower right corner please"],
}


def tokenize_instr(text: str) -> np.ndarray:
    """Tokenize an instruction into padded token ids (int32, MAX_INSTR_LEN)."""
    ids = [VOCAB_INDEX["<cls>"]] + [VOCAB_INDEX[w] for w in text.split() if w in VOCAB_INDEX]
    ids = ids[:MAX_INSTR_LEN]
    ids += [VOCAB_INDEX["<pad>"]] * (MAX_INSTR_LEN - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# MuJoCo model assembly
# ---------------------------------------------------------------------------
_SCENE_EXTRA = """
  <!-- pedestal raising the Franka base to table-top height -->
  <geom name="pedestal" type="cylinder" pos="0 0 0.30" size="0.12 0.30"
        contype="1" conaffinity="1" material="table_mat"/>
  <!-- table -->
  <body name="table" pos="{tx} {ty} {tz}">
    <geom name="table_top" type="box" size="0.30 0.30 0.02" condim="3"
          friction="1 0.005 0.0001" contype="3" conaffinity="3"
          material="table_mat"/>
    <geom name="table_leg1" type="box" pos="-0.24 -0.24 -0.30" size="0.03 0.03 0.28"
          contype="1" conaffinity="1" material="table_mat"/>
    <geom name="table_leg2" type="box" pos="0.24 -0.24 -0.30" size="0.03 0.03 0.28"
          contype="1" conaffinity="1" material="table_mat"/>
    <geom name="table_leg3" type="box" pos="-0.24 0.24 -0.30" size="0.03 0.03 0.28"
          contype="1" conaffinity="1" material="table_mat"/>
    <geom name="table_leg4" type="box" pos="0.24 0.24 -0.30" size="0.03 0.03 0.28"
          contype="1" conaffinity="1" material="table_mat"/>
    <body name="cam_tgt" pos="0 0 0.02"/>
  </body>
  <!-- target region markers (visual only, semi-transparent coloured pads) -->
  <body name="mkr_tl" pos="{mx_tl} {my_tl} {mz}">
    <geom type="box" size="0.020 0.020 0.004" rgba="1 0.2 0.2 0.55" contype="0" conaffinity="0"/>
  </body>
  <body name="mkr_tr" pos="{mx_tr} {my_tr} {mz}">
    <geom type="box" size="0.020 0.020 0.004" rgba="0.2 0.4 1 0.55" contype="0" conaffinity="0"/>
  </body>
  <body name="mkr_bl" pos="{mx_bl} {my_bl} {mz}">
    <geom type="box" size="0.020 0.020 0.004" rgba="1 0.8 0.2 0.55" contype="0" conaffinity="0"/>
  </body>
  <body name="mkr_br" pos="{mx_br} {my_br} {mz}">
    <geom type="box" size="0.020 0.020 0.004" rgba="1 0.5 0 0.55" contype="0" conaffinity="0"/>
  </body>
  <!-- cube (free joint, set at reset). Collision bitmasking: cube/tip use bit 2,
       table uses bits 1|2, arm links keep bit 1 -> the cube only ever contacts
       the tabletop and the fingertip, never the arm (panda-gym-like) -->
  <body name="cube" pos="{cx0} {cy0} {cz}">
    <freejoint/>
    <geom name="cube_geom" type="box" size="{ch} {ch} {ch}" condim="3"
          friction="1 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"
          contype="2" conaffinity="2" material="cube_mat"/>
  </body>
  <!-- camera looking at the table centre -->
  <camera name="cam" mode="targetbody" target="cam_tgt" pos="{cpx} {cpy} {cpz}" fovy="45"/>
"""
_SCENE_CONTACT = """
  <pair geom1="table_top" geom2="cube_geom"/>
  <pair geom1="tip_geom" geom2="cube_geom"/>
"""
_SCENE_ASSETS = """
    <material name="table_mat" rgba="0.55 0.55 0.58 1"/>
    <material name="cube_mat" rgba="0 0.85 0.2 1"/>
"""

_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])  # menagerie home:
# slightly bent, non-singular arm; EE ~ (0.555, 0, BASE_Z+0.624) -- beside the table


def _build_model(xml_path: str = PANDA_NO_HAND_XML):
    # meshdir="assets" in the menagerie XML is relative to the model dir
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(xml_path))
    try:
        with open(xml_path, encoding="utf-8") as fh:
            base = fh.read()
        extra = _SCENE_EXTRA.format(
            tx=TABLE_X, ty=TABLE_Y, tz=TABLE_TOP - 0.02,
            mx_tl=REGIONS["top-left"][0],     my_tl=REGIONS["top-left"][1],
            mx_tr=REGIONS["top-right"][0],    my_tr=REGIONS["top-right"][1],
            mx_bl=REGIONS["bottom-left"][0],  my_bl=REGIONS["bottom-left"][1],
            mx_br=REGIONS["bottom-right"][0], my_br=REGIONS["bottom-right"][1],
            mz=TABLE_TOP + 0.005,
            cx0=REGIONS["top-left"][0], cy0=REGIONS["bottom-left"][1], cz=CUBE_Z,
            ch=_CUBE_HALF,
            cpx=TABLE_X, cpy=TABLE_Y - 1.30, cpz=TABLE_TOP + 1.30,
        )
        # raise the whole Franka chain onto the pedestal, and turn the end-effector
        # site into a collision sphere (the fingertip) -- same position as the site,
        # so the observed EE pose and the pushing geometry coincide
        xml = base.replace('<body name="link0" childclass="panda">',
                           f'<body name="link0" childclass="panda" pos="0 0 {_BASE_Z}">', 1)
        xml = xml.replace('<body name="attachment" pos="0 0 0.107" quat="0.3826834 0 0 0.9238795">\n                      <site name="attachment_site"/>',
                          '<body name="attachment" pos="0 0 0.107" quat="0.3826834 0 0 0.9238795">\n'
                          '                      <geom name="tip_geom" type="sphere" size="0.022" condim="3"\n'
                          '                            friction="1 0.005 0.0001" solref="0.01 1"\n'
                          '                            contype="2" conaffinity="2" rgba="0.95 0.95 0.95 1"/>\n'
                          '                      <site name="attachment_site"/>', 1)
        xml = xml.replace("</worldbody>", extra + "</worldbody>")
        xml = xml.replace("</asset>", _SCENE_ASSETS + "</asset>", 1)
        xml = xml.replace("</contact>", _SCENE_CONTACT + "</contact>", 1) \
            if "</contact>" in xml else xml.replace(
                "</mujoco>", "<contact>" + _SCENE_CONTACT + "</contact></mujoco>", 1)
        model = mujoco.MjModel.from_xml_string(xml)
        return model
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
class LanguagePushEnv:
    """MuJoCo Franka push task wrapped for language-conditioned goal regions.

    API identical to vla_env.LanguagePushEnv:
      reset(region, ...) -> (rgb 64x64x3 uint8, instr int32[8], obs dict, info)
      step(action[3])    -> (rgb, reward, done, info{success, obs})
      obs dict: {observation: [ee_x, ee_y, ee_z, ...],
                 achieved_goal: cube pos, desired_goal: region centre}
    """

    IMG = 64

    def __init__(self, img: int = 64, max_episode_steps: int = 60, seed: int = 0,
                 act_scale: float = 0.2, sim_steps: int = 100, lam: float = 0.05,
                 dq_max: float = 0.03):
        self._model = _build_model()
        self._data = mujoco.MjData(self._model)
        self._site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._cube_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        assert self._site_id >= 0 and self._cube_id >= 0
        self._nv = self._model.nv
        self._qpos_cube0 = self._model.nq - 7  # freejoint starts here
        self._max_steps = max_episode_steps
        self.img = img
        self._rng = np.random.RandomState(seed)
        self._act_scale = act_scale
        self._sim_steps = sim_steps
        self._lam = lam
        self._dq_max = dq_max
        self._renderer = None
        self._t = 0

    @property
    def action_space(self):
        return None  # plain np array [-1,1]^3, as in the original wrapper

    # -- internals ----------------------------------------------------------
    def _render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, self.img, self.img)
        self._renderer.update_scene(self._data, camera="cam")
        return self._renderer.render().copy()

    def _ee_pos(self) -> np.ndarray:
        return self._data.site_xpos[self._site_id].copy()

    def _cube_pos(self) -> np.ndarray:
        return self._data.xpos[self._cube_id].copy()

    def _ik_to(self, target: np.ndarray, max_iter: int = 40, lam: float = 0.02,
               tol: float = 2e-3) -> np.ndarray:
        """Damped-least-squares IK: joint angles that put the EE site at target."""
        jacp = np.zeros((3, self._nv))
        jacr = np.zeros((3, self._nv))
        q = self._data.qpos[:7].copy()
        for _ in range(max_iter):
            self._data.qpos[:7] = q
            mujoco.mj_forward(self._model, self._data)
            e = target - self._data.site_xpos[self._site_id]
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
            J = jacp[:, :7]  # arm columns only (jacp includes the cube freejoint)
            A = J @ J.T + lam * np.eye(3)
            dq = J.T @ np.linalg.solve(A, e)
            q = q + dq
            # clamp to joint limits
            for j in range(7):
                lo, hi = self._model.jnt_range[j]
                q[j] = np.clip(q[j], lo, hi)
        return q

    # -- public API ---------------------------------------------------------
    def reset(self, region: str | int, start_min: float = 0.060,
              start_max: float = 0.085) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        if isinstance(region, int):
            region = REGION_NAMES[region]
        goal = REGIONS[region].copy()
        # cube start: uniform in the annulus around the region centre
        for _ in range(200):
            r = start_min + self._rng.uniform() * (start_max - start_min)
            a = self._rng.uniform(0.0, 2 * np.pi)
            xy = goal[:2] + np.array([r * np.cos(a), r * np.sin(a)])
            if abs(xy[0] - TABLE_X) <= 0.25 and abs(xy[1] - TABLE_Y) <= 0.25:
                break
        d = self._data
        d.qpos[:] = 0.0
        d.qpos[:7] = _HOME_QPOS
        d.qpos[self._qpos_cube0:self._qpos_cube0 + 3] = [xy[0], xy[1], CUBE_Z]
        d.qpos[self._qpos_cube0 + 3:self._qpos_cube0 + 7] = [1.0, 0.0, 0.0, 0.0]
        d.ctrl[:7] = _HOME_QPOS
        mujoco.mj_forward(self._model, d)
        # the cube is placed exactly resting on the table; no settle steps needed
        # (a settle loop made the arm drift into the table in early versions)
        self._t = 0
        rgb = self._render()
        instr = tokenize_instr(_TEMPLATES[region][0])
        obs = self._get_obs(goal)
        return rgb, instr, obs, {"region": region}

    def _get_obs(self, goal: np.ndarray) -> dict:
        ee = self._ee_pos()
        cube = self._cube_pos()
        return {
            "observation": np.concatenate([ee, cube]),
            "achieved_goal": cube,
            "desired_goal": goal.copy(),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        d = self._data
        # Move the EE by action * scale in world frame. Each physics substep:
        #  1) DLS in operational space resolves the remaining EE error into a
        #     small clamped joint increment (smooth trajectory near singularities),
        #  2) the position servo chases q_now + dq (real torques/contact dynamics,
        #     which is what actually pushes the cube).
        target = self._ee_pos() + action * self._act_scale
        jacp = np.zeros((3, self._nv)); jacr = np.zeros((3, self._nv))
        for _ in range(self._sim_steps):
            ee = self._data.site_xpos[self._site_id]
            err = target - ee
            if np.linalg.norm(err) < 2e-3:
                break
            mujoco.mj_jacSite(self._model, d, jacp, jacr, self._site_id)
            J = jacp[:, :7]
            A = J @ J.T + self._lam * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -self._dq_max, self._dq_max)
            d.ctrl[:7] = d.qpos[:7] + dq
            mujoco.mj_step(self._model, d)
        self._t += 1
        cube = self._cube_pos()
        # success criterion identical to the original: cube within 0.05 m (xy)
        # of the region centre
        # (goal is tracked on the wrapper from the reset call)
        success = self._goal is not None and bool(
            np.linalg.norm(cube[:2] - self._goal[:2]) < 0.05)
        rgb = self._render()
        obs = self._get_obs(self._goal)
        terminated = success
        truncated = self._t >= self._max_steps
        reward = -float(np.linalg.norm(cube[:2] - self._goal[:2])) if self._goal is not None else 0.0
        return rgb, reward, bool(terminated or truncated), {"success": success, "obs": obs}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# the reset() above stores the goal for step(): patch it in
_orig_reset = LanguagePushEnv.reset


def _reset_with_goal(self, region, start_min=0.060, start_max=0.085):
    out = _orig_reset(self, region, start_min, start_max)
    self._goal = REGIONS[region].copy() if not isinstance(region, int) else REGIONS[REGION_NAMES[region]].copy()
    self._region = region if isinstance(region, str) else REGION_NAMES[region]
    return out


LanguagePushEnv.reset = _reset_with_goal


# ---------------------------------------------------------------------------
# oracle: analytic pushing policy (identical logic to vla_env.make_oracle)
# ---------------------------------------------------------------------------
def make_oracle(grasp_gap: float = 0.09, ee_h: float = EE_H,
                k_approach: float = 8.0, push_speed: float = 0.85,
                z_gain: float = 4.0, fly_h: float = FLY_H,
                fly_high: float = FLY_H + 0.16):
    """Return a stateful analytic push policy: (obs) -> action.

    Phase A (fly): high-altitude transit to a point `grasp_gap` behind the cube
    (z = fly_high, well above the arm's swing volume, so the arm never sweeps
    the cube); Phase A' (approach): descend to fly_h just above the cube top;
    Phase B (lower): straight down onto the push height ee_h; Phase C (push):
    open-loop thrust along the recomputed cube->goal direction with z held;
    slow down to 0.35 when within 0.08 m of the goal. Re-approach once if the
    EE drifts > 0.14 m behind the cube.
    """
    state = {"phase": "fly"}

    def policy(obs: dict) -> np.ndarray:
        cube = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
        goal = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
        ee = np.asarray(obs["observation"][:3], dtype=np.float64).copy()

        to_goal = goal - cube
        to_goal[2] = 0.0
        dist_xy = np.linalg.norm(to_goal)
        if dist_xy > 1e-6:
            dirn = to_goal / dist_xy
        else:
            dirn = np.zeros(3)
            state["phase"] = "done"
        behind = cube - grasp_gap * dirn

        if state["phase"] == "fly":
            target = np.array([behind[0], behind[1], fly_high])
            to = target - ee
            if np.linalg.norm(to[:2]) < 0.02 and abs(to[2]) < 0.05:
                state["phase"] = "approach"
            else:
                return np.clip(k_approach * to, -1.0, 1.0).astype(np.float32)

        if state["phase"] == "approach":
            target = np.array([behind[0], behind[1], fly_h])
            to = target - ee
            if np.linalg.norm(to[:2]) < 0.02 and abs(to[2]) < 0.03:
                state["phase"] = "lower"
            else:
                return np.clip(k_approach * to, -1.0, 1.0).astype(np.float32)

        if state["phase"] == "lower":
            target = np.array([behind[0], behind[1], ee_h])
            to = target - ee
            # the fingertip sphere (r=0.022) enters the cube side band once its
            # bottom is below the cube top (0.68); no need to reach ee_h exactly
            if np.linalg.norm(to[:2]) < 0.025 and abs(to[2]) < 0.07:
                state["phase"] = "push"
            else:
                return np.clip(k_approach * to, -1.0, 1.0).astype(np.float32)

        if np.linalg.norm(ee[:2] - cube[:2]) > 0.14 and not state.get("reapproached"):
            state["reapproached"] = True
            state["phase"] = "fly"
            return policy(obs)
        speed = push_speed if dist_xy > 0.08 else 0.35
        act = np.array([speed * dirn[0], speed * dirn[1],
                        z_gain * (ee_h - ee[2])])
        return np.clip(act, -1.0, 1.0).astype(np.float32)

    return policy


# ---------------------------------------------------------------------------
# dataset collection (same protocol as vla_env.collect)
# ---------------------------------------------------------------------------
def collect(env: LanguagePushEnv, n_episodes: int, seed: int = 0,
            max_steps: int = 60, keep_only_success: bool = True) -> dict:
    rng = np.random.RandomState(seed)
    rgb_all, instr_all, action_all, region_all = [], [], [], []
    n_success = 0

    for ep in range(n_episodes):
        region = REGION_NAMES[rng.randint(len(REGION_NAMES))]
        rgb, instr, obs, _ = env.reset(region)
        tpls = [t for r, t in TEMPLATE_POOL if r == region]
        instr = tokenize_instr(tpls[rng.randint(len(tpls))])
        oracle = make_oracle()
        done = False
        steps = 0
        ep_rgb, ep_instr, ep_act = [], [], []
        while not done and steps < max_steps:
            act = oracle(obs)
            rgb, _, done, info = env.step(act)
            ep_rgb.append(rgb.transpose(2, 0, 1))  # (3,H,W) for torch
            ep_instr.append(instr)
            ep_act.append(act)
            obs = info["obs"]
            steps += 1
        if info["success"]:
            n_success += 1
            rgb_all += ep_rgb
            instr_all += ep_instr
            action_all += ep_act
            region_all += [REGION_NAMES.index(region)] * len(ep_act)
        if (ep + 1) % 100 == 0:
            print(f"  ep {ep + 1}/{n_episodes}  success_so_far={n_success}")

    n = len(action_all)
    print(f"oracle success: {n_success}/{n_episodes} ({n_success / n_episodes:.1%}); "
          f"kept {n} transition steps (success trajectories only)")
    return {
        "rgb": np.stack(rgb_all).astype(np.uint8),
        "instr": np.stack(instr_all).astype(np.int32),
        "action": np.stack(action_all).astype(np.float32),
        "region": np.array(region_all, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "verify-oracle", "collect"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data/vla_push_train.npz")
    ap.add_argument("--img", type=int, default=64)
    args = ap.parse_args()

    if args.cmd == "probe":
        env = LanguagePushEnv(img=args.img)
        rgb, instr, obs, info = env.reset("top-left")
        print("rgb", rgb.shape, rgb.dtype, "instr", instr.tolist(), "region", info["region"])
        print("ee", obs["observation"][:3], "cube", obs["achieved_goal"], "goal", obs["desired_goal"])
        t0 = time.time()
        for _ in range(10):
            rgb2, _, done, info = env.step(np.zeros(3, dtype=np.float32))
        dt = (time.time() - t0) / 10
        print(f"step+render {dt * 1000:.1f} ms/step -> ~{1 / dt:.0f} steps/s")
        env.close()

    elif args.cmd == "verify-oracle":
        env = LanguagePushEnv(img=args.img)
        n, ok, ep = 100, 0, 0
        rng = np.random.RandomState(0)
        while ep < n:
            region = REGION_NAMES[rng.randint(4)]
            rgb, instr, obs, _ = env.reset(region)
            oracle = make_oracle()
            for _ in range(60):
                act = oracle(obs)
                rgb, _, done, info = env.step(act)
                obs = info["obs"]
                if done:
                    break
            ok += int(info["success"])
            ep += 1
        print(f"oracle success {ok}/{n} ({ok / n:.1%})")
        env.close()

    elif args.cmd == "collect":
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        env = LanguagePushEnv(img=args.img)
        data = collect(env, n_episodes=args.n)
        env.close()
        np.savez_compressed(out, **data)
        for k, v in data.items():
            print(f"  {k}: {v.shape} {v.dtype}")
        print("saved ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
