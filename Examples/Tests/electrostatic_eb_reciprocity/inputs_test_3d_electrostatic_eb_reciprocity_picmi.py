#!/usr/bin/env python3
#
# Reciprocity of the electrostatic solve with an embedded boundary (3D): the
# potential at b of a charge at a must equal the potential at a of the same
# charge at b, also when a is next to the EB (see the RZ version of this
# test). A grounded conducting cylinder x^2 + y^2 < r_w^2 (cut cells; the
# wall cuts the nodes next to it in x and in y) in a periodic z domain, with
# point charges at rest at a (in the last cell before the wall) and b. Step 1
# with the charge at a gives G(b <- a) = phi_1(b); the same charge is then
# added at b and step 2 gives G(a <- b) = phi_2(a) - phi_1(a), with trilinear
# interpolation of phi, the transpose of the shape-1 deposit. The charge filter
# is off: filtering the deposit but not the gather breaks reciprocity by
# itself, and next to the wall the filter moves charge onto covered nodes.
# The results are written to reciprocity.json.

import argparse
import json

import numpy as np

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=16, help="cells per unit length")
parser.add_argument("--rw", type=float, default=0.837, help="wall radius")
parser.add_argument(
    "--a", type=float, nargs=3, default=[0.57, 0.57, 1.0], help="first charge"
)
parser.add_argument(
    "--b", type=float, nargs=3, default=[0.1, -0.2, 1.25], help="second charge"
)
parser.add_argument("--stencil", default=None, help="warpx.eb_poisson_stencil")
args, _ = parser.parse_known_args()

L, LZ = 1.0, 2.0
N, NZ = 2 * args.n, 2 * args.n
DX, DZ = 2 * L / N, LZ / NZ
W = 1.0e6  # weight of each particle
LO = np.array([-L, -L, 0.0])
DXYZ = np.array([DX, DX, DZ])

grid = picmi.Cartesian3DGrid(
    number_of_cells=[N, N, NZ],
    lower_bound=[-L, -L, 0.0],
    upper_bound=[L, L, LZ],
    lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    warpx_max_grid_size=16,
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
        implicit_function="sqrt(x*x + y*y) - rw", rw=args.rw
    ),
)
dist = picmi.ParticleListDistribution(
    x=[args.a[0]],
    y=[args.a[1]],
    z=[args.a[2]],
    ux=[0.0],
    uy=[0.0],
    uz=[0.0],
    weight=[W],
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
    phi = np.asarray(mf[:, :, :]).reshape(mf[:, :, :].shape[:3])
    s = (np.asarray(p) - LO) / DXYZ
    i = np.floor(s).astype(int)
    f = s - i
    v = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (f[0] if dx else 1 - f[0])
                    * (f[1] if dy else 1 - f[1])
                    * (f[2] if dz else 1 - f[2])
                )
                v += w * phi[i[0] + dx, i[1] + dy, i[2] + dz]
    return float(v)


sim.step(1)
phi1_a, phi1_b = phi_at(args.a), phi_at(args.b)
particle_containers.ParticleContainerWrapper("electrons").add_particles(
    x=[args.b[0]], y=[args.b[1]], z=[args.b[2]], ux=[0.0], uy=[0.0], uz=[0.0], w=[W]
)
sim.step(1)
g_ba = phi1_b
g_ab = phi_at(args.a) - phi1_a
out = {
    "stencil": args.stencil or "default",
    "n": args.n,
    "rw": args.rw,
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
