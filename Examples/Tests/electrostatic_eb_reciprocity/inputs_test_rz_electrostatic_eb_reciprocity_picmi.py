#!/usr/bin/env python3
#
# Reciprocity of the electrostatic solve with an embedded boundary (RZ): the
# potential at b of a charge at a must equal the potential at a of the same
# charge at b, also when a is next to the EB. Otherwise particles near the
# wall exert forces on the others that differ from the forces they feel, and
# the energy of a plasma is not conserved.
#
# A grounded conducting cylinder r < r_w (cut cells) in a periodic z domain.
# Ring charges at rest, at the points a (in the last cell before the wall, or
# on the node next to it) and b (inside). Step 1 with the charge at a gives
# phi_1, so G(b <- a) = phi_1(b); the same charge is then added at b, and
# step 2 gives G(a <- b) = phi_2(a) - phi_1(a). The potential at a point is
# the bilinear interpolation of phi on the nodes, the transpose of the
# shape-1 deposit. The charge filter is off (it is applied to the deposit
# only). The results are written to reciprocity.json.

import argparse
import json

import numpy as np

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=16, help="cells per unit length")
parser.add_argument("--rw", type=float, default=0.837, help="wall radius")
parser.add_argument(
    "--a", type=float, nargs=2, default=[0.8125, 1.0], help="r, z of the first charge"
)
parser.add_argument(
    "--b", type=float, nargs=2, default=[0.5, 1.25], help="r, z of the second charge"
)
parser.add_argument("--stencil", default=None, help="warpx.eb_poisson_stencil")
args, _ = parser.parse_known_args()

R, LZ = 1.0, 2.0
NR, NZ = args.n, 2 * args.n
DR, DZ = R / NR, LZ / NZ
W = 1.0e6  # weight of each ring

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R, LZ],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_max_grid_size=64,
    warpx_blocking_factor=4,
)
solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-13,
    warpx_absolute_tolerance=0.0,
)
sim = picmi.Simulation(
    solver=solver,
    time_step_size=1e-15,
    max_steps=2,
    verbose=0,
    particle_shape=1,
    warpx_use_filter=0,
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function="x - rw", rw=args.rw
    ),
)
dist = picmi.ParticleListDistribution(
    x=[args.a[0]], y=[0.0], z=[args.a[1]], ux=[0.0], uy=[0.0], uz=[0.0], weight=[W]
)
sim.add_species(
    picmi.Species(
        particle_type="electron", name="electrons", initial_distribution=dist
    ),
    layout=None,
)
sim.initialize_inputs()
if args.stencil is not None:
    import pywarpx

    pywarpx.warpx.eb_poisson_stencil = args.stencil
    pywarpx.warpx.eb_e_field = "centered"
sim.initialize_warpx()

from pywarpx import libwarpx, particle_containers  # noqa: E402

reg = libwarpx.libwarpx_so.get_instance().multifab_register()


def phi_at(p):
    mf = reg.get("phi_fp", level=0)
    phi = np.asarray(mf[:, :]).reshape(mf[:, :].shape[:2])
    i, j = int(p[0] / DR), int(p[1] / DZ)
    fr, fz = p[0] / DR - i, p[1] / DZ - j
    return float(
        (1 - fr) * (1 - fz) * phi[i, j]
        + fr * (1 - fz) * phi[i + 1, j]
        + (1 - fr) * fz * phi[i, j + 1]
        + fr * fz * phi[i + 1, j + 1]
    )


sim.step(1)
phi1_a, phi1_b = phi_at(args.a), phi_at(args.b)
particle_containers.ParticleContainerWrapper("electrons").add_particles(
    x=[args.b[0]], y=[0.0], z=[args.b[1]], ux=[0.0], uy=[0.0], uz=[0.0], w=[W]
)
sim.step(1)
g_ba = phi1_b
g_ab = phi_at(args.a) - phi1_a
h = (args.rw - DR * np.floor(args.rw / DR)) / DR
out = {
    "stencil": args.stencil or "default",
    "n": args.n,
    "rw": args.rw,
    "h": h,
    "a": args.a,
    "b": args.b,
    "G_b_from_a": g_ba,
    "G_a_from_b": g_ab,
    "ratio": g_ba / g_ab,
    "asymmetry": (g_ba - g_ab) / (0.5 * (g_ba + g_ab)),
}
print(json.dumps(out, indent=1))
with open("reciprocity.json", "w") as f:
    json.dump(out, f, indent=1)
