import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from figs import SURF, INK, INK2, GRID, C1, C2, C3, COVERED, OUT

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True,
                              gridspec_kw={"width_ratios": [1.35, 1]})
# left: 1D sketch, node i next to the wall, wall at h dx (h = 0.39)
h = 0.39
ax.set_xlim(-2.6, 1.9); ax.set_ylim(-1.75, 1.45); ax.axis("off")
ax.add_patch(Rectangle((h, -1.7), 1.5, 3.1, color=COVERED, lw=0))
ax.plot([h, h], [-1.7, 1.4], color=INK, lw=2)
ax.text(h + 0.75, 1.15, "conductor (EB)", ha="center", color=INK2, fontsize=10)
for x, lab, cov in ((-2, r"$i-2$", False), (-1, r"$b = i-1$", False), (0, r"$a = i$", False), (1, r"$i+1$", True)):
    ax.plot(x, 0, "o", ms=11, color=INK2 if cov else C1, mfc=SURF if cov else C1, mew=2)
    ax.text(x, -0.38, lab, ha="center", va="top", fontsize=11)
ax.annotate("", (h, -1.3), (0, -1.3), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
ax.text(h / 2 + 0.05, -1.36, r"$h\,\Delta x$", ha="center", va="top", fontsize=11, color=INK2)
arc = dict(arrowstyle="-|>", mutation_scale=14, lw=1.6)
ax.add_patch(FancyArrowPatch((-0.08, 0.12), (-0.92, 0.12), connectionstyle="arc3,rad=0.55", color=C2, **arc))
ax.add_patch(FancyArrowPatch((-0.92, -0.12), (-0.08, -0.12), connectionstyle="arc3,rad=0.55", color=C3, **arc))
ax.text(-0.75, 0.8, r"row $a$:  $A_{a,b} = \dfrac{2}{1+h}\,\dfrac{1}{\Delta x^2}$", ha="center", fontsize=12, color=INK)
ax.text(-0.75, -0.85, r"row $b$:  $A_{b,a} = \dfrac{1}{\Delta x^2}$", ha="center", fontsize=12, color=INK)
ax.set_title("Shortley-Weller: the elements between $a$ and $b$ differ by $2/(1+h)$", fontsize=10.5)
# right: measured ratio against h
hh = np.linspace(0, 1, 101)
ax2.plot(hh, (1 + hh) / 2, color=C2, ls="--", label=r"$(1+h)/2$")
ax2.plot([0.048, 0.392, 0.920], [0.5240, 0.6960, 0.9600], "s", color=C2, ms=8, label="Shortley-Weller, measured")
ax2.plot([0.048, 0.392, 0.920], [1, 1, 1], "^", color=C3, ms=8, label=r"symmetric, measured ($1 \pm 10^{-12}$)")
ax2.set_xlim(0, 1); ax2.set_ylim(0.4, 1.08)
ax2.set_xlabel(r"$h$ (cut-edge length of the node $a$ next to the wall)")
ax2.set_ylabel(r"$G(b \leftarrow a)\,/\,G(a \leftarrow b)$")
ax2.set_title("reciprocity ratio, RZ cylinder", fontsize=10.5)
ax2.legend(loc="lower right", fontsize=9)
fig.savefig(OUT / "reciprocity.png", dpi=150)
