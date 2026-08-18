"""Batched BP-SDF as torch.nn.Module for L4CasADi export.

Input layout matches GCDF online packing (12-D):
  [x, y, z, qx, qy, yaw, j1, j2, j3, j4, j5, j6]
where (x,y) are usually obstacle XY relative to base, qx=qy=0.
Output: (B, 1) whole-body signed distance (meters).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BP_PATH = os.path.abspath(
    os.path.join(CUR_DIR, "../../gcdf-training/resource/models/BP_8.pt")
)

# Serial chain for Kinova Gen3 6-DoF + yaw (matches moma.urdf / MoMaLayer_pk link order).
# Each entry: (child_link_name, rpy, xyz, joint_index or None if fixed)
# joint_index indexes into theta = q[2:9] (yaw + 6 arm joints).
_CHAIN: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], int | None]] = [
    ("base_link", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0),  # joint_yaw
    ("armbase_link", (0.0, 0.0, 0.0), (0.03385, -0.00764, 0.306), None),  # arm_to_chassis
    ("armshoulder_link", (-math.pi, 0.0, 0.0), (0.0, 0.0, 0.15643), 1),
    ("armbicep_link", (math.pi / 2, 0.0, 0.0), (0.0, 0.005375, -0.12838), 2),
    ("armforearm_link", (math.pi, 0.0, 0.0), (0.0, -0.41, 0.0), 3),
    ("armspherical_wrist_1_link", (math.pi / 2, 0.0, 0.0), (0.0, 0.20843, -0.006375), 4),
    ("armspherical_wrist_2_link", (-math.pi / 2, 0.0, 0.0), (0.0, -0.00017505, -0.10593), 5),
    ("armbracelet_link", (math.pi / 2, 0.0, 0.0), (0.0, 0.10593, -0.00017505), 6),
]

# Links that have BP-SDF weights (same as bf_sdf reorder ∩ used_links[0:8]).
_SDF_LINKS = [
    "base_link",
    "armbase_link",
    "armshoulder_link",
    "armbicep_link",
    "armforearm_link",
    "armspherical_wrist_1_link",
    "armspherical_wrist_2_link",
    "armbracelet_link",
]


def _rpy_xyz_matrix(rpy: Sequence[float], xyz: Sequence[float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # ZYX intrinsic / URDF rpy
    R = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
    return T


def _rotz_batch(theta: torch.Tensor) -> torch.Tensor:
    """theta: (B,) -> (B, 4, 4) rotation about z. Out-of-place for vmap/L4CasADi."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    row0 = torch.stack([c, -s, z, z], dim=-1)
    row1 = torch.stack([s, c, z, z], dim=-1)
    row2 = torch.stack([z, z, o, z], dim=-1)
    row3 = torch.stack([z, z, z, o], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


class BPSDFNet(nn.Module):
    def __init__(
        self,
        bp_model_path: str = DEFAULT_BP_PATH,
        n_func: int = 8,
        domain_min: float = -1.0,
        domain_max: float = 1.0,
    ):
        super().__init__()
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max

        raw = torch.load(bp_model_path, map_location="cpu", weights_only=False)
        by_name: Dict[str, dict] = {v["mesh_name"]: v for v in raw.values()}

        weights = []
        offsets = []
        scales = []
        for name in _SDF_LINKS:
            item = by_name[name]
            weights.append(item["weights"].float().reshape(-1))
            offsets.append(item["offset"].float().reshape(3))
            scales.append(float(item["scale"]))
        self.register_buffer("weights", torch.stack(weights, dim=0))  # (K, n_func^3)
        self.register_buffer("offsets", torch.stack(offsets, dim=0))  # (K, 3)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))  # (K,)

        fixed = [_rpy_xyz_matrix(rpy, xyz) for _, rpy, xyz, _ in _CHAIN]
        self.register_buffer("fixed_T", torch.stack(fixed, dim=0))  # (L, 4, 4)
        # Python constants only — required for L4CasADi / make_fx tracing.
        self._joint_idx_py = [(-1 if j is None else int(j)) for *_, j in _CHAIN]
        sdf_link_set = set(_SDF_LINKS)
        self._sdf_chain_index_py = [
            i for i, (name, *_) in enumerate(_CHAIN) if name in sdf_link_set
        ]

        # Bernstein binomial coefficients C(n, i)
        n = n_func - 1
        i = torch.arange(n_func, dtype=torch.float32)
        log_comb = (
            torch.lgamma(torch.tensor(n + 1.0))
            - torch.lgamma(i + 1.0)
            - torch.lgamma(torch.tensor(n + 1.0) - i)
        )
        self.register_buffer("binom", torch.exp(log_comb))
        self.register_buffer(
            "bernstein_i", torch.arange(n_func, dtype=torch.float32)
        )

    def _bernstein_1d(self, t: torch.Tensor) -> torch.Tensor:
        """t: (..., n_coords) in [0,1] -> (..., n_coords, n_func)."""
        t = torch.clamp(t, 1e-4, 1.0 - 1e-4)
        n = self.n_func - 1
        i = self.bernstein_i.to(dtype=t.dtype, device=t.device)
        comb = self.binom.to(dtype=t.dtype, device=t.device)
        t = t.unsqueeze(-1)
        return comb * (1.0 - t) ** (n - i) * t ** i

    def build_basis(self, p: torch.Tensor) -> torch.Tensor:
        """p: (M, 3) in domain -> (M, n_func^3)."""
        p_norm = (p - self.domain_min) / (self.domain_max - self.domain_min)
        phi = self._bernstein_1d(p_norm)
        phi_x, phi_y, phi_z = phi[:, 0, :], phi[:, 1, :], phi[:, 2, :]
        phi_xy = torch.einsum("bi,bj->bij", phi_x, phi_y).reshape(p.shape[0], -1)
        phi_xyz = torch.einsum("bi,bj->bij", phi_xy, phi_z).reshape(p.shape[0], -1)
        return phi_xyz

    def fk_link_transforms(self, theta: torch.Tensor) -> torch.Tensor:
        """theta: (B, 7) -> (B, K, 4, 4) transforms for SDF links."""
        B = theta.shape[0]
        T = (
            torch.eye(4, device=theta.device, dtype=theta.dtype)
            .unsqueeze(0)
            .expand(B, 4, 4)
            .contiguous()
        )
        link_Ts = []
        for i, j in enumerate(self._joint_idx_py):
            T_fixed = self.fixed_T[i].to(dtype=theta.dtype)
            T = torch.matmul(T, T_fixed.unsqueeze(0).expand(B, 4, 4))
            if j >= 0:
                T = torch.matmul(T, _rotz_batch(theta[:, j]))
            link_Ts.append(T)
        stacked = torch.stack(link_Ts, dim=1)  # (B, L, 4, 4)
        return stacked[:, self._sdf_chain_index_py, :, :]

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        inp: (B, 12) = [x(3), q(9)]
        returns: (B, 1) SDF
        """
        if inp.dim() != 2 or inp.shape[-1] != 12:
            raise ValueError(f"Expected (B,12) input, got {tuple(inp.shape)}")

        x = inp[:, 0:3]
        qxy = inp[:, 3:5]
        theta = inp[:, 5:12]
        B = inp.shape[0]
        K = self.weights.shape[0]

        Ts = self.fk_link_transforms(theta)  # (B, K, 4, 4)
        R = Ts[:, :, :3, :3]
        t = Ts[:, :, :3, 3]
        # Apply base XY in the point frame: x' = x - [qx, qy, 0]
        x_shifted = torch.stack(
            [x[:, 0] - qxy[:, 0], x[:, 1] - qxy[:, 1], x[:, 2]], dim=-1
        )
        x_link = torch.matmul(
            R.transpose(-1, -2), (x_shifted.unsqueeze(1) - t).unsqueeze(-1)
        ).squeeze(-1)  # (B, K, 3)

        scale = self.scales.to(dtype=inp.dtype)
        offset = self.offsets.to(dtype=inp.dtype)
        x_scaled = (x_link - offset.unsqueeze(0)) / scale.view(1, K, 1)

        x_bounded = torch.clamp(x_scaled, -1.0 + 1e-2, 1.0 - 1e-2)
        res = x_scaled - x_bounded

        phi = self.build_basis(x_bounded.reshape(B * K, 3)).reshape(B, K, -1)
        w = self.weights.to(dtype=inp.dtype)
        sdf = torch.einsum("bkn,kn->bk", phi, w) + res.norm(dim=-1)
        sdf = sdf * scale.view(1, K)
        return sdf.min(dim=1).values.unsqueeze(-1)


def build_bpsdf_net(bp_model_path: str = DEFAULT_BP_PATH, device: str = "cuda") -> BPSDFNet:
    net = BPSDFNet(bp_model_path=bp_model_path)
    return net.to(device).eval()
