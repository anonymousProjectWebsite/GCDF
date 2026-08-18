"""Export BP-SDF through L4CasADi with the same CasADi API shape as GCDF.

Produces:
  ../core/bpsdf_func.c / .h
  ./_l4c_generated/bpsdf*

Symbols (same names as GCDF so ConstraintFunction can swap .so later):
  y(x)                  : (12, N) -> (N, 1)
  df_test_truncated(x)  : (12, N) -> (N, 9*N)  jacobian w.r.t. q only
"""

from __future__ import annotations

import argparse
import os
import sys

import casadi as cs
import l4casadi as l4c
import torch

from bpsdf_net import DEFAULT_BP_PATH, build_bpsdf_net

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def verify_against_reference(net: torch.nn.Module, device: str, n: int = 32) -> None:
    """Compare BPSDFNet FK/SDF with MoMaLayer_pk + training BPSDF (lazy import)."""
    import types

    train_root = os.path.abspath(os.path.join(CUR_DIR, "../../gcdf-training"))
    sys.path.insert(0, train_root)
    sys.path.insert(0, os.path.join(train_root, "resource"))

    # bf_sdf pulls optional viz deps; stub anything missing for query-only use.
    for mod_name in ("mesh_to_sdf", "skimage", "trimesh"):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = types.ModuleType(mod_name)

    import bf_sdf
    from robot_layer.moma_layer_pk import MoMaLayer

    robot = MoMaLayer(device)
    bp = bf_sdf.BPSDF(8, -1.0, 1.0, robot, device)
    bp_model = torch.load(
        os.path.join(train_root, "resource/models/BP_8.pt"),
        map_location=device,
        weights_only=False,
    )

    torch.manual_seed(0)
    x = (torch.rand(n, 3, device=device) - 0.5) * 2.0
    x[:, 2] = torch.rand(n, device=device) * 1.5
    q = robot.q_min + torch.rand(n, 9, device=device) * (robot.q_max - robot.q_min)
    theta = q[:, 2:]
    pose = torch.eye(4, device=device).unsqueeze(0).expand(n, 4, 4).float()

    # --- FK check (first 8 pk links == SDF links) ---
    pk_list = robot.get_transformations_each_link(pose, theta)
    for i in range(len(pk_list)):
        pk_list[i] = pk_list[i].clone()
        pk_list[i][:, 0, 3] += q[:, 0]
        pk_list[i][:, 1, 3] += q[:, 1]
    T_pk = torch.stack(pk_list[:8], dim=1)
    with torch.no_grad():
        T_ours = net.fk_link_transforms(theta)
        T_ours = T_ours.clone()
        T_ours[:, :, 0, 3] += q[:, 0].unsqueeze(-1)
        T_ours[:, :, 1, 3] += q[:, 1].unsqueeze(-1)
    fk_err = (T_ours - T_pk).abs().max().item()
    print(f"[verify] FK max abs err vs pk: {fk_err:.6e}")
    if fk_err > 1e-4:
        raise RuntimeError("FK mismatch vs MoMaLayer_pk")

    ref = []
    for i in range(n):
        sdf_i, _ = bp.get_whole_body_sdf_batch_xy(
            x[i : i + 1],
            pose[i : i + 1],
            q[i : i + 1, 2:],
            q[i : i + 1, :2],
            bp_model,
            use_derivative=False,
        )
        ref.append(sdf_i.reshape(-1))
    ref = torch.cat(ref, dim=0)

    inp = torch.cat([x, q], dim=-1)
    with torch.no_grad():
        pred = net(inp).reshape(-1)

    err = (pred - ref).abs()
    print(
        f"[verify] SDF n={n} MAE={err.mean().item():.6f} "
        f"max={err.max().item():.6f} median={err.median().item():.6f}"
    )
    if err.mean().item() > 5e-3:
        raise RuntimeError("BPSDFNet differs from reference BP-SDF too much")


def export_l4casadi(net: torch.nn.Module, batch_size: int, device: str, name: str = "bpsdf") -> None:
    l4c_model = l4c.L4CasADi(net, device=device, batched=True, name=name)
    num_input = 12
    x_sym = cs.SX.sym("x", num_input, batch_size)
    y_sym = l4c_model(x_sym.T)
    f = cs.Function("y", [x_sym], [y_sym])

    jac_y_x = cs.jacobian(y_sym, x_sym)
    x_sym_truncated = cs.vertcat(
        x_sym[3, :],
        x_sym[4, :],
        x_sym[5, :],
        x_sym[6, :],
        x_sym[7, :],
        x_sym[8, :],
        x_sym[9, :],
        x_sym[10, :],
        x_sym[11, :],
    )  # (9, N)
    x_sym_truncated_flat = x_sym_truncated.reshape((-1, 1))
    x_sym_flat = x_sym.reshape((-1, 1))
    jac_trunc_x = cs.jacobian(x_sym_truncated_flat, x_sym_flat)
    jac_y_trunc = jac_y_x @ jac_trunc_x.T
    df_truncated = cs.Function("df_test_truncated", [x_sym], [jac_y_trunc])

    prefix_code = os.path.abspath(os.path.join(CUR_DIR, "../core/"))
    os.makedirs(prefix_code, exist_ok=True)
    opts = {"cpp": False, "with_header": True}
    cg = cs.CodeGenerator("bpsdf_func.c", opts)
    cg.add(f)
    cg.add(df_truncated)
    cg.generate(os.path.join(prefix_code, ""))
    print(f"Generated {prefix_code}/bpsdf_func.c and header")
    print(f"L4CasADi generated lib dir: {l4c_model.shared_lib_dir}")


def smoke_torch(net: torch.nn.Module, batch_size: int, device: str) -> None:
    x = torch.randn(batch_size, 12, device=device)
    x[:, 3:5] = 0.0
    with torch.no_grad():
        y = net(x)
    print(f"[smoke] torch forward shape={tuple(y.shape)} mean={y.mean().item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp-model", default=DEFAULT_BP_PATH)
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--save-pth", default=os.path.join(CUR_DIR, "model_bpsdf.pth"))
    args = parser.parse_args()

    print("casadi:", cs.__file__)
    print("device:", args.device, "batch_size:", args.batch_size)

    net = build_bpsdf_net(args.bp_model, device=args.device)
    torch.save(net, args.save_pth)
    print("saved", args.save_pth)

    if not args.skip_verify:
        verify_against_reference(net, args.device)
    if args.verify_only:
        return

    smoke_torch(net, min(args.batch_size, 64), args.device)
    export_l4casadi(net, args.batch_size, args.device)


if __name__ == "__main__":
    main()
