"""Grasp model: per-finger contact, hand state (none / touch / hold), vertex contact, MANO mesh.

Per frame:
  fingertip branch (= TouchNet's: 5 aligned fingertip patches -> encoder -> proj, landmark MLP)
  hand branch: axis-aligned hand crop (hand + nearby object) -> encoder -> hand embedding
Temporal (causal GRUs; step t sees frames <= t only):
  finger GRU (shared over the 5 fingers): [tip emb, lm emb, present, hand context] -> contact logit
  hand GRU: [hand emb, the 5 finger-GRU outputs, present] -> state logits [3], vertex logits [778]
Mesh (frame t only, no temporal): [hand features, landmark features] -> 16 joint rotations (6D,
  Zhou et al. CVPR 2019) + 10 shape betas -> MANO (right hand; left hands are mirrored in the
  crop and the labels) -> root-relative vertices [778, 3] and joints [21, 3] (metres, camera axes).
init_from_touchnet copies a trained TouchNet into the fingertip branch; the new GRU input columns
(hand context) start at zero, so at step 0 the finger output equals the TouchNet's.

bidir=True: OFFLINE variant for labelling recorded demonstrations (not for live use). Both GRUs are
bidirectional and frame t is scored from a window centred on it (window_target = T // 2), so it
sees future frames. init_from_causal copies a trained causal model in, with the backward readout
zeroed, so at step 0 it equals the causal model run on the past half of the window.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from touch.data.patches import LM_FEAT_DIMS
from touch.mano_path import MANO_DIR  # noqa: F401  (re-exported)

TIPS_RIGHT = [745, 317, 444, 556, 673]
ORDER21 = [0, 13, 14, 15, -1, 1, 2, 3, -1, 4, 5, 6, -1, 10, 11, 12, -1, 7, 8, 9, -1]


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """[..., 6] -> [..., 3, 3] (Gram-Schmidt on the two predicted columns)."""
    a, b = x[..., :3], x[..., 3:]
    c1 = nn.functional.normalize(a, dim=-1)
    c2 = nn.functional.normalize(b - (c1 * b).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], -1)


class ManoLayer(nn.Module):
    """Right-hand MANO (flat hand mean) from rotation matrices, via smplx's lbs."""

    def __init__(self, model_dir: Path = MANO_DIR):
        super().__init__()
        import smplx

        m = smplx.MANO(model_path=str(model_dir), is_rhand=True, use_pca=False,
                       flat_hand_mean=True)  # fmt: skip
        for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "lbs_weights"):
            self.register_buffer(k, getattr(m, k).detach().clone().float(), persistent=False)
        self.register_buffer("parents", m.parents.detach().clone(), persistent=False)

    def forward(self, rotmats: torch.Tensor, betas: torch.Tensor):
        """rotmats [B, 16, 3, 3], betas [B, 10] -> root-relative verts [B, 778, 3] and
        joints [B, 21, 3] (metres)."""
        from smplx.lbs import lbs

        v, j16 = lbs(betas, rotmats, self.v_template, self.shapedirs, self.posedirs,
                     self.J_regressor, self.parents, self.lbs_weights, pose2rot=False)  # fmt: skip
        j21 = torch.stack(
            [v[:, TIPS_RIGHT[i // 4 - 1]] if k < 0 else j16[:, k] for i, k in enumerate(ORDER21)], 1
        )
        root = j21[:, :1]
        return v - root, j21 - root


def _encoder(name: str, pretrained: bool, size: int) -> tuple[nn.Module, int]:
    enc = timm.create_model(name, pretrained=pretrained, num_classes=0)
    with torch.no_grad():
        w = enc.eval()(torch.zeros(1, 3, size, size)).shape[-1]
    enc.train()
    return enc, w


class GraspNet(nn.Module):
    def __init__(
        self,
        tip_backbone: str = "mobilenetv3_large_100.ra_in1k",
        hand_backbone: str = "repvit_m1_1.dist_450e_in1k",
        pretrained: bool = True,
        n_lm: int = LM_FEAT_DIMS["v2"],
        emb: int = 128,
        lm_emb: int = 64,
        hidden: int = 128,
        hand_emb: int = 256,
        ctx: int = 64,
        hand_hidden: int = 256,
        dropout: float = 0.2,
        img_drop: float = 0.2,
        hand_drop: float = 0.1,
        mesh: bool = True,
        grad_frames: int = 0,
        tip_size: int = 112,
        hand_size: int = 160,
        bidir: bool = False,
        table: bool = False,
    ):
        super().__init__()
        self.causal = not bidir
        nd = 2 if bidir else 1
        self.n_lm, self.img_drop, self.hand_drop = n_lm, img_drop, hand_drop
        self.tip_size, self.hand_size = tip_size, hand_size
        self.grad_frames = grad_frames  # training: encoder gradients from the newest k frames only
        # fingertip branch: names match TouchNet (encoder, proj, lm_mlp, gru, head)
        self.encoder, wt = _encoder(tip_backbone, pretrained, tip_size)
        self.proj = nn.Sequential(nn.Linear(wt, emb), nn.ReLU(), nn.Dropout(dropout))
        self.lm_mlp = nn.Sequential(
            nn.Linear(n_lm + 5, 2 * lm_emb), nn.ReLU(), nn.Linear(2 * lm_emb, lm_emb), nn.ReLU()
        )
        self.hand_encoder, wh = _encoder(hand_backbone, pretrained, hand_size)
        self.hand_proj = nn.Sequential(nn.Linear(wh, hand_emb), nn.ReLU(), nn.Dropout(dropout))
        self.ctx = nn.Sequential(nn.Linear(hand_emb, ctx), nn.ReLU())
        self.d_touch = emb + lm_emb + 1  # TouchNet's per-finger input width
        self.gru = nn.GRU(self.d_touch + ctx, hidden, batch_first=True, bidirectional=bidir)
        self.head = nn.Linear(nd * hidden, 1)
        self.hand_gru = nn.GRU(hand_emb + 5 * nd * hidden + 1, hand_hidden, batch_first=True,
                               bidirectional=bidir)  # fmt: skip
        self.state_head = nn.Linear(nd * hand_hidden, 3)
        self.vert_head = nn.Linear(nd * hand_hidden, 778)
        self.use_table = table  # "hand touches the table" logit (scripts/make_table_labels.py)
        if table:
            self.table_head = nn.Linear(nd * hand_hidden, 1)
        self.use_mesh = mesh
        if mesh:
            self.mesh_mlp = nn.Sequential(
                nn.Linear(wh + n_lm, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 16 * 6 + 10),
            )
            last = self.mesh_mlp[-1]
            nn.init.normal_(last.weight, std=1e-3)
            with torch.no_grad():  # start at the identity rotation, mean shape
                last.bias.zero_()
                last.bias[: 16 * 6].copy_(torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(16))
            self.mano = ManoLayer()

    def window_target(self, T: int) -> int:
        """Window position of the frame being scored: newest (causal) or centre (bidir)."""
        return T - 1 if self.causal else T // 2

    def encode(self, tips, hand, lm, present) -> dict:
        """tips [B, T, 5, 3, s, s], hand [B, T, 3, S, S], lm [B, T, F], present [B, T]
        -> per-frame features (no temporal mixing)."""
        B, T = present.shape
        e = self.proj(self._frames(self.encoder, tips)).view(B, T, 5, -1)
        hf = self._frames(self.hand_encoder, hand)
        he = self.hand_proj(hf)
        if self.training and self.img_drop > 0:
            e = e * (torch.rand(B, 1, 1, 1, device=e.device) >= self.img_drop).to(e.dtype)
        if self.training and self.hand_drop > 0:
            he = he * (torch.rand(B, 1, 1, device=e.device) >= self.hand_drop).to(e.dtype)
        oh = torch.eye(5, device=lm.device).expand(B, T, 5, 5)
        lm5 = torch.cat([lm[:, :, None].expand(B, T, 5, lm.shape[-1]), oh], -1)
        le = self.lm_mlp(lm5) * present[..., None, None]
        p = present[..., None, None].float().expand(B, T, 5, 1)
        c = self.ctx(he)[:, :, None].expand(B, T, 5, -1)
        tok = torch.cat([e, le, p, c], -1)
        return {"tok": tok, "hand_emb": he, "hand_feat": hf, "lm": lm, "present": present}

    def _frames(self, enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run an image encoder over x [B, T, ..., C, H, W] -> [B, T, ..., W]. While training
        with grad_frames = k > 0, only k frames get encoder gradients: the newest k (causal) or
        the k around the window centre (bidir); the rest are encoded without gradients (saves
        activation memory; the GRUs still see every frame)."""
        lead = x.shape[:-3]
        T, k = lead[1], self.grad_frames
        if not self.training or k <= 0 or k >= T:
            return enc(x.flatten(0, len(lead) - 1)).view(*lead, -1)
        a = T - k if self.causal else max(0, min(T - k, self.window_target(T) - k // 2))

        def run(sl: slice) -> torch.Tensor:
            y = x[:, sl]
            return enc(y.flatten(0, len(lead) - 1)).view(lead[0], y.shape[1], *lead[2:], -1)

        parts = []
        with torch.no_grad():
            if a > 0:
                parts.append((0, run(slice(0, a))))
            if a + k < T:
                parts.append((2, run(slice(a + k, T))))
        parts.append((1, run(slice(a, a + k))))
        return torch.cat([p for _, p in sorted(parts, key=lambda q: q[0])], 1)

    def temporal(self, enc: dict) -> dict:
        tok, he, present = enc["tok"], enc["hand_emb"], enc["present"]
        B, T = present.shape
        h, _ = self.gru(tok.permute(0, 2, 1, 3).reshape(B * 5, T, -1))
        finger = self.head(h).view(B, 5, T).transpose(1, 2)
        hin = torch.cat(
            [he, h.view(B, 5, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1),
             present[..., None].float()], -1
        )  # fmt: skip
        g, _ = self.hand_gru(hin)
        out = {"finger": finger, "state": self.state_head(g), "vert": self.vert_head(g)}
        if self.use_table:
            out["table"] = self.table_head(g)[..., 0]
        return out

    def mesh(self, hand_feat: torch.Tensor, lm: torch.Tensor) -> dict:
        """hand_feat [B, W], lm [B, F] (one frame) -> verts [B, 778, 3], joints [B, 21, 3],
        betas [B, 10], rot [B, 16, 3, 3] (MANO joint rotations, right hand, flat hand mean)."""
        out = self.mesh_mlp(torch.cat([hand_feat, lm], -1))
        R = rot6d_to_matrix(out[:, : 16 * 6].reshape(-1, 16, 6))
        betas = out[:, 16 * 6 :]
        v, j = self.mano(R, betas)
        return {"verts": v, "joints": j, "betas": betas, "rot": R}

    def forward(self, tips, hand, lm, present, with_mesh: bool = True) -> dict:
        enc = self.encode(tips, hand, lm, present)
        out = self.temporal(enc)
        if with_mesh and self.use_mesh:
            t = self.window_target(present.shape[1])
            out.update(self.mesh(enc["hand_feat"][:, t], enc["lm"][:, t]))
        return out


def init_from_touchnet(model: GraspNet, ckpt_path: str | Path) -> list[str]:
    """Copy a TouchNet checkpoint's fingertip branch into a GraspNet (same backbone, emb, lm_emb,
    hidden). GRU input weights get zero columns for the new hand-context inputs."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ck["model"]
    dst = model.state_dict()
    copied = []
    for k, v in src.items():
        if k not in dst:
            continue
        if k == "gru.weight_ih_l0":
            w = torch.zeros_like(dst[k])
            assert v.shape[1] == model.d_touch, (v.shape, model.d_touch)
            w[:, : v.shape[1]] = v
            dst[k] = w
        elif dst[k].shape == v.shape:
            dst[k] = v
        else:
            continue
        copied.append(k)
    model.load_state_dict(dst)
    return copied


def init_from_causal(model: GraspNet, ckpt_path: str | Path) -> list[str]:
    """Copy a trained causal GraspNet into a bidir one (same sizes). Forward-direction GRU weights,
    encoders and the mesh head copy as is; backward-direction GRU weights keep their random init
    (so they get gradients) while every input/output column that reads the backward direction
    starts at zero, so the bidir model initially equals the causal model."""
    src = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    assert not model.causal
    dst = model.state_dict()
    H = model.gru.hidden_size
    copied = []
    for k, v in src.items():
        if k not in dst:
            continue
        d = dst[k]
        if d.shape == v.shape:
            dst[k] = v
        elif k in ("head.weight", "state_head.weight", "vert_head.weight"):
            w = torch.zeros_like(d)
            w[:, : v.shape[1]] = v  # [forward | backward] features
            dst[k] = w
        elif k == "hand_gru.weight_ih_l0":
            w = torch.zeros_like(d)
            E = model.hand_proj[0].out_features
            w[:, :E] = v[:, :E]
            for f in range(5):  # finger outputs: old [f*H..], new [f*2H .. f*2H+H] (forward half)
                w[:, E + f * 2 * H : E + f * 2 * H + H] = v[:, E + f * H : E + (f + 1) * H]
            w[:, -1] = v[:, -1]  # present
            dst[k] = w
        else:
            continue
        copied.append(k)
    model.load_state_dict(dst)
    return copied


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


class GraspStep(nn.Module):
    """One live frame (causal). Inputs: tips [5, 3, s, s], hand [3, S, S] (normalized), lm [F],
    present [] bool, hist_tok [T-1, 5, d], hist_he [T-1, E], hist_present [T-1] (the previous
    T-1 frames' encodings, oldest first). Outputs: finger [5], state [3], vert [778] probabilities,
    verts [778, 3], joints [21, 3] (root-relative, right-hand canonical), tok [5, d], he [E] (append
    to the history), and table [] probability last when the model has a table head.
    Re-running the GRUs over the window from a zero state equals evaluation."""

    def __init__(self, model: GraspNet):
        super().__init__()
        assert model.causal, "GraspStep is the live (causal) model"
        self.m = model

    def forward(self, tips, hand, lm, present, hist_tok, hist_he, hist_present):
        e = self.m.encode(tips[None, None], hand[None, None], lm[None, None], present[None, None])
        win = {
            "tok": torch.cat([hist_tok[None], e["tok"]], 1),
            "hand_emb": torch.cat([hist_he[None], e["hand_emb"]], 1),
            "present": torch.cat([hist_present[None], present[None, None]], 1),
        }
        out = self.m.temporal(win)
        mesh = self.m.mesh(e["hand_feat"][:, 0], lm[None])
        outs = (
            torch.sigmoid(out["finger"][0, -1]),
            torch.softmax(out["state"][0, -1], -1),
            torch.sigmoid(out["vert"][0, -1]),
            mesh["verts"][0],
            mesh["joints"][0],
            e["tok"][0, 0],
            e["hand_emb"][0, 0],
        )
        if self.m.use_table:  # appended last: older consumers keep their output order
            outs = (*outs, torch.sigmoid(out["table"][0, -1]))
        return (*outs, mesh["rot"][0], mesh["betas"][0])  # MANO params [16, 3, 3], [10]


@torch.no_grad()
def pad_encoding(model: GraspNet) -> dict:
    """Encoding of a frame before the sequence start (black images, no hand), as in training."""
    from touch.data.grasp_store import to_input

    tips = to_input(torch.zeros(1, 1, 5, model.tip_size, model.tip_size, 3, dtype=torch.uint8))
    hand = to_input(torch.zeros(1, 1, model.hand_size, model.hand_size, 3, dtype=torch.uint8))
    e = model.encode(tips, hand, torch.zeros(1, 1, model.n_lm), torch.zeros(1, 1, dtype=torch.bool))
    return {"tok": e["tok"][0, 0], "hand_emb": e["hand_emb"][0, 0]}
