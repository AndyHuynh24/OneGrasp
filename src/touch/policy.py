"""A robot policy trained from the demonstrations in Tiger Data (behaviour cloning).

Observation (37): joint_pos 7, joint_vel 7, gripper_qpos 2, eef_pos 3, eef_quat 4 (xyzw),
cup_pos 3, cup_quat 4 (xyzw), goal 6 (the cup's place position xy + its final orientation as
quat), last gripper command 1.
  The goal is where and how the demo left the cup (its last recorded cup pose): the same for
  teleop and generated demos.
Action: a chunk of the next K steps: joint targets as offsets from the current joint position
(7) + gripper (closed logit) -- executed with temporal ensembling (ACT, Zhao et al. 2023:
every step predicts a chunk; the overlapping predictions for a step are averaged with
weights exp(-m * age)). Controller = the recorder's (touch.panda.simulate).

Success in simulation (the rule demos were kept by, touch.demogen.check, written against the
recorded goal): lifted >= MIN_LIFT, final cup xy within PLACE_TOL of the goal, cup axis
within TILT_MAX of the goal's axis (a lying cup's axis either way), resting on the table,
gripper open.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn

from touch.demogen import MIN_LIFT, PLACE_TOL, TILT_MAX
from touch.panda import SUB

OBS_DIM, ACT_DIM = 37, 8
GRIP_OPEN = 255.0  # the recorder's open command for this cup (open width capped at 8 cm)


def goal_of(a: dict) -> np.ndarray:
    """[6]: the demo's last cup xy + last cup quat (xyzw, w >= 0)."""
    q = np.asarray(a["cup_quat"][-1], np.float64)
    q = q if q[3] >= 0 else -q
    return np.r_[np.asarray(a["cup_pos"][-1], np.float64)[:2], q]


def obs_of(a: dict, t: int, goal: np.ndarray, grip_prev: float) -> np.ndarray:
    return np.concatenate([a["joint_pos"][t], a["joint_vel"][t], a["gripper_qpos"][t],
                           a["eef_pos"][t], a["eef_quat"][t], a["cup_pos"][t], a["cup_quat"][t],
                           goal, [grip_prev]]).astype(np.float32)  # fmt: skip


def demo_samples(a: dict, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One demo -> obs [T, 37], action chunks [T, K, 8] (joint offset 7 + gripper 0/1; past the
    end the last action repeats), and a mask [T, K] (1 = inside the demo)."""
    T = len(a["action"])
    goal = goal_of(a)
    grip = (a["action"][:, -1] > 0).astype(np.float32)
    X = np.stack([obs_of(a, t, goal, grip[t - 1] if t else grip[0]) for t in range(T)])
    idx = np.minimum(np.arange(T)[:, None] + np.arange(K)[None], T - 1)  # [T, K]
    dq = a["action"][idx, :7] - a["joint_pos"][:, None, :]
    A = np.concatenate([dq, grip[idx][..., None]], -1).astype(np.float32)
    M = (np.arange(T)[:, None] + np.arange(K)[None] < T).astype(np.float32)
    return X, A, M


class Policy(nn.Module):
    """MLP: normalised obs [B, 37] -> chunk [B, K, 8] (7 normalised joint offsets + 1 logit)."""

    def __init__(self, K: int = 16, hidden: int = 512, layers: int = 4, use_vel: bool = False):
        super().__init__()
        self.K = K
        # joint velocities off by default: with them the policy copies the arm's current speed
        # (causal confusion / copycat: de Haan et al. 2019, Wen et al. 2020) and, starting at
        # rest, barely moves (measured 2026-11-14: 0/34 val scenes lifted with them)
        self.register_buffer("vel_on", torch.tensor(float(use_vel)))
        dims = [OBS_DIM] + [hidden] * layers
        net = []
        for a, b in zip(dims, dims[1:], strict=False):
            net += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.body = nn.Sequential(*net)
        self.head = nn.Linear(hidden, K * ACT_DIM)
        for k in ("o_mu", "o_sd", "a_mu", "a_sd"):
            self.register_buffer(k, torch.zeros(OBS_DIM if k[0] == "o" else 7))

    def set_norm(self, X: np.ndarray, A: np.ndarray) -> None:
        self.o_mu.copy_(torch.as_tensor(X.mean(0)))
        self.o_sd.copy_(torch.as_tensor(X.std(0) + 1e-3))
        dq = A[..., :7].reshape(-1, 7)
        self.a_mu.copy_(torch.as_tensor(dq.mean(0)))
        self.a_sd.copy_(torch.as_tensor(dq.std(0) + 1e-4))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.o_mu) / self.o_sd
        if not bool(self.vel_on):
            x = torch.cat([x[:, :7], torch.zeros_like(x[:, 7:14]), x[:, 14:]], 1)
        x = torch.cat([x[:, :-1], torch.zeros_like(x[:, -1:])], 1)  # last gripper command:
        # off (copycat too: the policy repeated it and never closed, 2026-11-14)
        return self.head(self.body(x)).view(-1, self.K, ACT_DIM)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One obs [37] -> (joint offsets [K, 7] in rad, P(gripper closed) [K])."""
        y = self(torch.as_tensor(obs[None]))[0]
        dq = y[:, :7] * self.a_sd + self.a_mu
        return dq.numpy(), torch.sigmoid(y[:, 7]).numpy()


def loss_fn(pol: Policy, X, A, M) -> torch.Tensor:
    y = pol(X)
    dq = (A[..., :7] - pol.a_mu) / pol.a_sd
    l_q = (torch.abs(y[..., :7] - dq).mean(-1) * M).sum() / M.sum()
    l_g = (nn.functional.binary_cross_entropy_with_logits(y[..., 7], A[..., 7], reduction="none")
           * M).sum() / M.sum()  # fmt: skip
    return l_q + 0.5 * l_g


def _obs_sim(m, d, ids: dict, goal, grip_prev: float) -> np.ndarray:
    """The observation from a running simulation (the recorder's quantities)."""
    q = d.qpos[ids["qadr"]]
    qd = d.qvel[ids["vadr"]]
    fing = d.qpos[ids["fin"]]
    tcp = d.site_xpos[ids["sid"]]
    tq = Rot.from_matrix(d.site_xmat[ids["sid"]].reshape(3, 3)).as_quat()
    cup = d.xpos[ids["obj"]]
    cq = d.xquat[ids["obj"]][[1, 2, 3, 0]]
    return np.concatenate([q, qd, fing, tcp, tq, cup, cq, goal, [grip_prev]]).astype(np.float32)


def rollout(pol: Policy, xml: str, state0, goal, n_steps: int, ens_m: float = 0.1,
            render=None) -> dict:  # fmt: skip
    """Closed loop in MuJoCo from a recorded first state -> {"success", "why", metrics, ...}.
    render(m, d) is called each step (optional)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = state0[: m.nq]
    d.qvel[:] = state0[m.nq : m.nq + m.nv]
    mujoco.mj_forward(m, d)
    ids = {"qadr": np.array([m.jnt_qposadr[m.joint(f"joint{k}").id] for k in range(1, 8)]),
           "vadr": np.array([m.jnt_dofadr[m.joint(f"joint{k}").id] for k in range(1, 8)]),
           "fin": np.array([m.jnt_qposadr[m.joint(f"finger_joint{k}").id] for k in (1, 2)]),
           "sid": m.site("tcp").id,
           "obj": m.body("obj0").id}  # fmt: skip
    K = pol.K
    buf_q = np.zeros((n_steps + K, K, 7))  # predictions for step s made at step s - j
    buf_g = np.zeros((n_steps + K, K))
    have = np.zeros((n_steps + K, K), bool)
    dt = SUB * m.opt.timestep
    q_prev = d.qpos[ids["qadr"]].copy()
    grip_prev = 0.0
    z = []
    for t in range(n_steps):
        obs = _obs_sim(m, d, ids, goal, grip_prev)
        dq, pg = pol.act(obs)
        q_now = d.qpos[ids["qadr"]].copy()
        for j in range(K):
            buf_q[t + j, j] = q_now + dq[j]
            buf_g[t + j, j] = pg[j]
            have[t + j, j] = True
        w = np.exp(-ens_m * np.arange(K))[have[t]]  # oldest prediction weighted most (ACT)
        q = (buf_q[t][have[t]] * w[:, None]).sum(0) / w.sum()
        closed = float((buf_g[t][have[t]] * w).sum() / w.sum()) > 0.5
        grip_prev = float(closed)
        d.ctrl[7] = 0.0 if closed else GRIP_OPEN
        qd = (q - q_prev) / dt
        for s in range(SUB):
            d.ctrl[:7] = q_prev + (q - q_prev) * (s + 1) / SUB + 0.1 * qd
            mujoco.mj_step(m, d)
        q_prev = q
        z.append(float(d.xpos[ids["obj"]][2]))
        if render is not None:
            render(m, d)
    return judge(m, d, ids, goal, np.asarray(z), _cup_z0(m, state0), grip_prev < 0.5)


def _cup_z0(m, state0) -> float:
    """The cup's height in the first state (its free joint's z)."""
    j = m.body("obj0").jntadr[0]
    return float(state0[m.jnt_qposadr[j] + 2])


def judge(m, d, ids, goal, z: np.ndarray, z0: float, grip_open: bool) -> dict:
    """Success against the recorded goal (module docstring)."""
    gs = m.geom_size[m.geom("cyl").id]  # the demo's own object (radius, half height)
    cup_d, cup_h = 2 * float(gs[0]), 2 * float(gs[1])
    end = d.xpos[ids["obj"]]
    ax = Rot.from_quat(d.xquat[ids["obj"]][[1, 2, 3, 0]]).as_matrix()[:, 2]
    gax = Rot.from_quat(goal[2:]).as_matrix()[:, 2]
    lying = abs(gax[2]) < 0.5
    cosang = abs(ax @ gax) if lying else ax @ gax
    tilt = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    err = float(np.linalg.norm(end[:2] - goal[:2]))
    lift = float(z.max() - z0) if len(z) else 0.0
    rest = abs(float(end[2]) - (cup_d / 2 if lying else cup_h / 2)) < 0.01
    released = grip_open and float(d.qpos[ids["fin"]].sum()) > cup_d + 0.004
    met = {"lift_cm": round(100 * lift, 1), "place_err_cm": round(100 * err, 1),
           "axis_err_deg": round(tilt, 1), "released": bool(released)}  # fmt: skip
    for bad, why in ((lift < MIN_LIFT, "not lifted"), (tilt > TILT_MAX, "wrong orientation"),
                     (err > PLACE_TOL, f"missed by {100 * err:.1f} cm"),
                     (not rest, "not resting on the table"),
                     (not released, "not released")):  # fmt: skip
        if bad:
            return {"success": False, "why": why, **met}
    return {"success": True, "why": "ok", **met}
