#!/usr/bin/env python3
# Figures for the EB electrostatics PRs (novatron-fusion/WarpX#9, novatron-fusion/amrex#1, #2).
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

SP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SP / "nw"))
from proto import Geo  # noqa: E402

OUT = Path(__file__).resolve().parent
# reference palette, light mode
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3aeae", "#e34948", "#8c1f1f"])
COVERED = "#d9d8d3"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.8, "lines.markersize": 5,
})
A, B, V, VO = 0.3117, 0.8233, 50.0, -20.0
K = (V - VO) / (1 / A - 1 / B)


def nearwall_errors(fn):
    d = np.load(fn)
    p = d["p"]
    n = int(d["n"])
    rho = np.linalg.norm(p, axis=1)
    outer = rho > 0.5 * (A + B)
    t = np.where(outer, B - rho, rho - A) * n
    ex = K * p / rho[:, None] ** 3
    err = np.linalg.norm(d["Eg"] - ex, axis=1) / np.linalg.norm(ex, axis=1)
    return p, t, err, outer


METHODS = [("SW + gradient", C1, "-", "o"), ("conforming, first layer only", C2, "--", "s"),
           ("conforming, all layers", C3, "-", "^")]
NW = SP / "nw"
FILES = {1: ["nw64_shortley_weller_gradient.npz", "nw64_symmetric_conforming.npz", "nw64s1_G.npz"],
         3: ["nw64s3_shortley_weller_gradient.npz", "nw64s3_old.npz", "nw64s3_G.npz"]}


def fig_nearwall_dist():
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    edges = np.linspace(0, 1.5, 16)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for row, shape in enumerate((1, 3)):
        ax0, ax1 = axs[row]
        for (label, col, ls, mk), fn in zip(METHODS, FILES[shape]):
            _, t, err, _ = nearwall_errors(NW / fn)
            e = np.sort(err)
            frac = 1.0 - np.arange(len(e)) / len(e)
            ax0.step(e, frac, where="post", color=col, ls=ls, label=label)
            mx = [err[(t >= a) & (t < b)].max() for a, b in zip(edges[:-1], edges[1:])]
            ax1.plot(mid, mx, color=col, ls=ls, marker=mk, markersize=5, label=label)
        name = "linear" if shape == 1 else "cubic"
        ax0.set_xscale("log"); ax0.set_yscale("log")
        ax0.set_xlim(1e-4, 1.2); ax0.set_ylim(2e-4, 1.05)
        ax0.set_xlabel("relative error of the gathered field |E − E_exact| / |E_exact|")
        ax0.set_ylabel("fraction of particles with a larger error")
        ax0.set_title(f"{name} particles: distribution")
        ax1.set_yscale("log"); ax1.set_ylim(3e-3, 1.2); ax1.set_xlim(0, 1.5)
        ax1.set_xlabel("distance of the particle to the wall (cells)")
        ax1.set_ylabel("largest relative error in the bin")
        ax1.set_title(f"{name} particles: largest error vs distance")
    axs[0][0].legend(loc="lower left")
    fig.suptitle("Field gathered by 4000 test particles within 1.5 cells of the spheres\n"
                 "of the spherical capacitor (3D, n = 64; symmetric stencil with conforming)", fontsize=10.5, color=INK2)
    fig.savefig(OUT / "nearwall_error_distribution.png", dpi=150)


def fig_nearwall_map():
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.2), constrained_layout=True)
    norm = LogNorm(1e-3, 1.0)
    for row, shape in enumerate((1, 3)):
        for colix, ((label, _, _, _), fn) in enumerate(zip(METHODS, FILES[shape])):
            ax = axs[row][colix]
            p, t, err, outer = nearwall_errors(NW / fn)
            m = (~outer) & (p[:, 2] > 0)          # inner sphere, upper half, seen from +z
            o = np.argsort(err[m])                 # draw the large errors on top
            sc = ax.scatter(p[m, 0][o], p[m, 1][o], c=err[m][o], cmap=SEQ, norm=norm, s=14,
                            edgecolors="none")
            ax.add_patch(plt.Circle((0, 0), A, fill=False, color=INK2, lw=0.8))
            ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3]); ax.set_yticks([-0.3, -0.15, 0, 0.15, 0.3])
            ax.set_xlim(-0.34, 0.34); ax.set_ylim(-0.34, 0.34); ax.set_aspect("equal")
            ax.set_title(f"{'linear' if shape == 1 else 'cubic'} particles\n{label}\n"
                         f"max {err[m].max() * 100:.0f} %, 95th pct. {np.percentile(err[m], 95) * 100:.1f} %",
                         fontsize=9.5)
            ax.set_xlabel("x (m)")
            if colix == 0:
                ax.set_ylabel("y (m)")
            ax.tick_params(labelsize=8)
    cb = fig.colorbar(sc, ax=axs, shrink=0.6, location="right")
    cb.set_label("relative error of the gathered field")
    fig.suptitle("Test particles within 1.5 cells of the inner sphere (radius 0.3117 m, upper half, seen from +z),\n"
                 "coloured by the relative error of the field they gather; 3D, n = 64", fontsize=10.5, color=INK2)
    fig.savefig(OUT / "nearwall_error_map.png", dpi=150)


CONV = SP / "conv"


def sph(n, st):
    d = np.load(CONV / f"sph_rz_{n}_{st}.npz")
    err = np.where(d["gap"], (d["phi"] - d["ex"]) / 70.0, np.nan)
    return d, err


def fig_sphcap_maps():
    n = 64
    d, es = sph(n, "shortley_weller")
    _, ey = sph(n, "symmetric")
    r = d["r"][:, 0]; z = d["z"][0, :] - 1
    fig, axs = plt.subplots(1, 3, figsize=(12, 6.4), constrained_layout=True)
    phi = np.where(d["gap"], d["phi"], np.nan)
    ax = axs[0]
    ax.set_facecolor(COVERED)
    im = ax.pcolormesh(r, z, phi.T, cmap=SEQ, shading="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="φ (V)")
    ax.set_title("potential φ\n(symmetric stencil)")
    lim = np.nanmax(np.abs(ey)) * 1e3
    norm = TwoSlopeNorm(0.0, -lim, lim)
    for ax, e, name in ((axs[1], es, "Shortley-Weller"), (axs[2], ey, "symmetric")):
        ax.set_facecolor(COVERED)
        im = ax.pcolormesh(r, z, (e * 1e3).T, cmap=DIV, norm=norm, shading="nearest")
        ax.set_title(f"{name}: (φ − φ_exact) / 70 V\nmax |…| = {np.nanmax(np.abs(e)) * 1e3:.2f}e-3")
    fig.colorbar(im, ax=axs[1:], shrink=0.7, label="(φ − φ_exact) / 70 V  (×10⁻³)")
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel("r (m)"); ax.grid(False)
        for R in (A, B):
            th = np.linspace(-np.pi / 2, np.pi / 2, 200)
            ax.plot(R * np.cos(th), R * np.sin(th), color=INK2, lw=0.7)
    axs[0].set_ylabel("z − 1 (m)")
    fig.suptitle("Spherical capacitor in RZ, n = 64: spheres of radius 0.3117 m at 50 V and 0.8233 m at −20 V;\n"
                 "covered nodes in gray", fontsize=10.5, color=INK2)
    fig.savefig(OUT / "sphcap_rz_maps.png", dpi=150)


def fig_sphcap_conv():
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axs[0]
    for st, label, col, ls, mk in (("shortley_weller", "Shortley-Weller", C1, "-", "o"),
                                   ("symmetric", "symmetric", C2, "-", "s")):
        mx = [np.nanmax(np.abs(sph(n, st)[1])) * n * n for n in ns]
        ax.plot(ns, mx, color=col, ls=ls, marker=mk, label=label)
    ax.axhline(1 / (4 * A**3 * (1 / A - 1 / B)), color=INK2, ls=":", lw=1.2)
    ax.annotate("flat-wall estimate, h = 1/2", (16, 4.25), color=INK2, fontsize=9)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 4.8)
    ax.set_xlabel("n (cells per unit length)"); ax.set_ylabel("max |φ − φ_exact| / 70 V  × n²")
    ax.set_title("maximum error × n² (constant = second order)")
    ax.legend(loc="center right")
    ax = axs[1]
    n = 512
    for st, label, col, mk in (("shortley_weller", "Shortley-Weller", C1, "o"), ("symmetric", "symmetric", C2, "s")):
        d, e = sph(n, st)
        r, zz = d["r"], d["z"] - 1
        rho = np.hypot(r, zz)
        m = (np.abs(rho - A) < 3 / n) & np.isfinite(e)
        ang = np.degrees(np.arctan2(r, np.abs(zz)))[m]
        ax.scatter(ang, np.abs(e[m]) * n * n, s=4, color=col, alpha=0.6, edgecolors="none", label=label)
    ax.set_xlim(0, 90); ax.set_ylim(0, 4.5)
    ax.set_xlabel("angle from the axis (degrees; 0 = pole, 90 = equator)")
    ax.set_ylabel("|φ − φ_exact| / 70 V  × n²")
    ax.set_title("error at the nodes within 3 cells of the inner sphere, n = 512")
    ax.legend(loc="upper right", markerscale=3)
    fig.suptitle("Spherical capacitor in RZ", fontsize=10.5, color=INK2)
    fig.savefig(OUT / "sphcap_rz_convergence.png", dpi=150)


WN = SP / "wneu"


def fig_neumann():
    X0 = 0.61803
    n = 64
    res = {}
    for tag in ("before", "after"):
        h = np.load(WN / f"{tag}_plane_3d_n64_shortley_weller_gradient_0.npz")
        f = np.load(WN / f"{tag}_plane_3d_n64_shortley_weller_gradient_1.npz")
        ph = h["phi"].reshape(h["phi"].shape[:3]); pf = f["phi"].reshape(f["phi"].shape[:3])[:, :, n:]
        res[tag] = (ph, ph - pf)
    x = np.arange(n + 1) / n
    X, Z = np.meshgrid(x, x, indexing="ij")
    unc = X - X0 + Z < 0                           # uncovered for x < x0 - z, z >= 0 (any y)
    unc3 = np.broadcast_to(unc[:, None, :], (n + 1, n + 1, n + 1))
    ph, dfb = res["before"]
    dmask = np.where(unc3, dfb, 0.0)
    j0 = n // 2                                    # slice y = 0.5 m
    lim = np.abs(dmask).max()
    norm = LogNorm(1e-6, lim)
    fig = plt.figure(figsize=(13, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(COVERED)
    im = ax.pcolormesh(x, x, np.where(unc, res["after"][0][:, j0, :], np.nan).T, cmap=SEQ, shading="nearest")
    fig.colorbar(im, ax=ax, shrink=0.75, label="φ (V)")
    ax.set_title(f"φ with the fix\nslice y = {j0 / n:.2f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)  (Neumann face at z = 0)")
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(COVERED)
    ax2.pcolormesh(x, x, np.where(unc, np.maximum(np.abs(dfb[:, j0, :]), 1e-12), np.nan).T, cmap=SEQ, norm=norm,
                   shading="nearest")
    ax2.set_title(f"|φ_half − φ_full| without the fix\nsame slice, zoomed")
    ax2.set_xlim(0.35, 0.7); ax2.set_ylim(0, 0.35)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)")
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor(COVERED)
    face = np.where((x < X0)[:, None], np.maximum(np.abs(dfb[:, :, 0]), 1e-12), np.nan)
    im3 = ax3.pcolormesh(x, x, face.T, cmap=SEQ, norm=norm, shading="nearest")
    fig.colorbar(im3, ax=[ax2, ax3], shrink=0.75, label="|φ_half − φ_full| (V), without the fix")
    ax3.set_title("|φ_half − φ_full| without the fix\non the Neumann face z = 0")
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
    for a in (ax, ax2, ax3):
        a.set_aspect("equal"); a.grid(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    for a in (ax, ax2):
        a.plot([X0, X0 - 1], [0, 1], color=INK2, lw=0.7)
    ax3.axvline(X0, color=INK2, lw=0.7)
    ax4 = fig.add_subplot(gs[3])
    v = np.sort(np.abs(dfb[unc3])); v = v[v > 0]
    frac = 1.0 - np.arange(len(v)) / len(v)
    ax4.step(v, frac * len(v) / unc3.sum(), where="post", color=C2, ls="--", label="without the fix")
    amax = np.abs(res["after"][1][unc3]).max()
    ax4.axvline(amax, color=C3, lw=1.8, label=f"with the fix: max {amax:.0e} V")
    ax4.set_xscale("log"); ax4.set_yscale("log"); ax4.set_ylim(1e-5, 1.1); ax4.set_xlim(1e-17, 1)
    ax4.set_xlabel("|φ_half − φ_full| (V)")
    ax4.set_ylabel("fraction of uncovered nodes above")
    ax4.set_title("distribution over all\nuncovered nodes")
    ax4.legend(loc="lower left", fontsize=9)
    fig.suptitle("EB plane at 45° to a Neumann face (z = 0): half domain z ≥ 0 against the domain mirrored about "
                 "z = 0\n3D, n = 64, Shortley-Weller stencil; EB at 1 + x/2 V, the other faces at 0 V",
                 fontsize=10.5, color=INK2)
    fig.savefig(OUT / "neumann_face.png", dpi=150)


if __name__ == "__main__":
    fig_nearwall_dist()
    fig_nearwall_map()
    fig_sphcap_maps()
    fig_sphcap_conv()
    fig_neumann()
    print("written", sorted(p.name for p in OUT.glob("*.png")))
