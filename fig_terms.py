import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from figs import SURF, INK, INK2, GRID, C1, C2, C3, COVERED, OUT

NX, NY = 8, 6
XC, YC, RC = 9.3, 7.6, 6.0          # the conductor: inside this circle (upper right)
X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
cov = np.hypot(X - XC, Y - YC) < RC

def hcut(i, j, di, dj):
    """uncovered length from node (i, j) toward its covered neighbour (i+di, j+dj)"""
    p = np.array([i - XC, j - YC], float); d = np.array([di, dj], float)
    b = 2 * p @ d; c = p @ p - RC * RC
    return (-b - np.sqrt(b * b - 4 * c)) / 2

def base(ax, title):
    ax.set_xlim(1.5, NX - 0.4); ax.set_ylim(1.5, NY - 0.4); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((XC, YC), RC, color=COVERED, lw=0, zorder=0))
    th = np.linspace(np.pi, 1.5 * np.pi, 200)
    ax.plot(XC + RC * np.cos(th), YC + RC * np.sin(th), color=INK, lw=2, zorder=1)
    for i in range(NX):
        ax.plot([i, i], [0, NY - 1], color=GRID, lw=1, zorder=0.5)
    for j in range(NY):
        ax.plot([0, NX - 1], [j, j], color=GRID, lw=1, zorder=0.5)
    ax.set_title(title, fontsize=10.5, loc="left")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.2), constrained_layout=True)
# (a) terms
base(a1, "(a) covered and uncovered nodes, cut edges, $h^-$ and $h^+$")
for i in range(NX):
    for j in range(NY):
        for di, dj in ((1, 0), (0, 1)):
            ii, jj = i + di, j + dj
            if ii < NX and jj < NY and cov[i, j] != cov[ii, jj]:
                a1.plot([i, ii], [j, jj], color=C2, lw=3, zorder=1.5)
for i in range(NX):
    for j in range(NY):
        a1.plot(i, j, "o", ms=9, zorder=3, color=INK2 if cov[i, j] else C1, mfc=SURF if cov[i, j] else C1, mew=1.8)
# the node shown: uncovered, with covered neighbours at +x and +y
P = next((i, j) for i in range(NX) for j in range(NY)
         if not cov[i, j] and i + 1 < NX and j + 1 < NY and cov[i + 1, j] and cov[i, j + 1] and i > 0 and j > 0)
i, j = P
hx, hy = hcut(i, j, 1, 0), hcut(i, j, 0, 1)
a1.plot(i, j, "o", ms=15, mfc="none", mec=INK, mew=1.5, zorder=4)
# EB crossings on the cut edges, and the four lengths of node P
for ii in range(NX):
    for jj in range(NY):
        for di, dj in ((1, 0), (0, 1)):
            i2, j2 = ii + di, jj + dj
            if i2 < NX and j2 < NY and cov[ii, jj] != cov[i2, j2]:
                u = (ii, jj) if not cov[ii, jj] else (i2, j2)
                sgn = 1 if u == (ii, jj) else -1
                t = hcut(u[0], u[1], sgn * di, sgn * dj)
                a1.plot(u[0] + sgn * di * t, u[1] + sgn * dj * t, "o", ms=5, color=INK, zorder=3.5)
lab = dict(fontsize=11, zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"))
a1.text(i - 0.5, j + 0.1, r"$h^-_x = 1$", ha="center", va="bottom", **lab)
nodelab = dict(fontsize=13, zorder=6, color=INK)
a1.text(i - 0.12, j - 0.1, r"$\mathbf{a}$", ha="right", va="top", **nodelab)
a1.text(i - 1 - 0.12, j - 0.1, r"$\mathbf{b}$", ha="right", va="top", **nodelab)
a1.text(i + hx / 2, j - 0.1, r"$h^+_x = h$", ha="center", va="top", **lab)
a1.text(i - 0.08, j - 0.5, r"$h^-_y = 1$", ha="right", va="center", **lab)
a1.text(i - 0.08, j + hy / 2, r"$h^+_y$", ha="right", va="center", **lab)
a1.annotate(r"$h\,\Delta x$: distance from" "\n" r"the node to the EB", (i + hx, j), xytext=(i + 1.3, j - 0.85),
            fontsize=10, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), zorder=5)
a1.plot([], [], "o", color=C1, ms=9, label="uncovered node")
a1.plot([], [], "o", color=INK2, mfc=SURF, mew=1.8, ms=9, label="covered node (in the conductor)")
a1.plot([], [], color=C2, lw=3, label="cut edge")
a1.plot([], [], color=INK, lw=2, label="EB (conductor surface)")
a1.plot([], [], "o", color=INK, ms=5, label="where the EB crosses an edge")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)
# (b) conforming ghost values, linear particles (D = 1)
base(a2, "(b) $\\mathtt{conforming}$, linear particles: covered nodes with ghost values")
unc = ~cov
lay = np.where(unc, -1, 0)
for i in range(NX):
    for j in range(NY):
        if cov[i, j] and any(0 <= i + di < NX and 0 <= j + dj < NY and unc[i + di, j + dj]
                             for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            lay[i, j] = 1
src = {}
for l in (2, 3):
    add = []
    for i in range(NX):
        for j in range(NY):
            if lay[i, j] != 0:
                continue
            near = any(0 <= i + di < NX and 0 <= j + dj < NY and unc[i + di, j + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1))
            if not near:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = (i + di, j + dj), (i + 2 * di, j + 2 * dj)
                if all(0 <= q[0] < NX and 0 <= q[1] < NY for q in (a, b)) and 0 < lay[a] < l and (lay[b] == -1 or 0 < lay[b] < l):
                    add.append((i, j)); src[(i, j)] = (a, b); break
    for q in add:
        lay[q] = l
cols = {-1: C1, 0: None, 1: C2, 2: C3, 3: C3}
for i in range(NX):
    for j in range(NY):
        l = lay[i, j]
        if l == -1:
            a2.plot(i, j, "o", ms=9, color=C1, zorder=3)
        elif l == 0:
            a2.plot(i, j, "o", ms=9, color=INK2, mfc=SURF, mew=1.8, zorder=3)
        else:
            a2.plot(i, j, "s" if l == 1 else "D", ms=10, color=cols[l], zorder=3)
# one second-layer node: its extrapolation
q = sorted(src)[len(src) // 2]
a, b = src[q]
a2.plot([b[0], a[0], q[0]], [b[1], a[1], q[1]], color=INK, lw=1.4, ls="--", zorder=2)
a2.annotate(r"$\phi = 2\phi_1 - \phi_2$", (q[0], q[1]), xytext=(q[0] + 0.25, q[1] + 0.35), fontsize=10.5, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=GRID))
a2.text(a[0] + 0.12, a[1] + 0.12, r"$\phi_1$", fontsize=10); a2.text(b[0] + 0.12, b[1] + 0.12, r"$\phi_2$", fontsize=10)
a2.plot([], [], "o", color=C1, ms=9, label="uncovered node")
a2.plot([], [], "s", color=C2, ms=9, label="first layer: next to an uncovered node along an edge")
a2.plot([], [], "D", color=C3, ms=9, label="later layers: other corners of cells with an uncovered corner")
a2.plot([], [], "o", color=INK2, mfc=SURF, mew=1.8, ms=9, label="covered node without a ghost value")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=9)
fig.savefig(OUT / "terms.png", dpi=150)
print("node", P, "h_x", round(hx, 2), "h_y", round(hy, 2), "layers", {l: int((lay == l).sum()) for l in (1, 2, 3)})
