#!/usr/bin/env python3
#
# Field near an embedded boundary (RZ, electrostatic): a spherical capacitor,
# the region between two concentric spheres of radius a (at V) and b (at V_o),
# centred on the axis. Test particles of negligible weight at random positions
# within 1.5 cells of either sphere gather E after one step; the gathered field
# is compared with the exact field K (r, z - z_c) / rho^3, K = (V - V_o) /
# (1/a - 1/b). Records the largest relative error of the particles closer to
# the wall than 0.6 cells and of those farther away (warpx.eb_e_field,
# warpx.eb_poisson_stencil, particle shape). Writes nearwall.json.

import argparse
import json

import numpy as np
from scipy.constants import e, m_e

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=32, help="cells per unit length")
parser.add_argument("--np", type=int, default=1000, help="test particles")
parser.add_argument("--shape", type=int, default=1, help="particle shape")
parser.add_argument("--eb-e-field", default="conforming")
parser.add_argument("--eb-poisson-stencil", default="symmetric")
args, _ = parser.parse_known_args()

A, B, V, VO = 0.3117, 0.8233, 50.0, -20.0
ZC = 1.0
K = (V - VO) / (1 / A - 1 / B)
DX = 1.0 / args.n

grid = picmi.CylindricalGrid(
    number_of_cells=[args.n, 2 * args.n],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[1.0, 2.0],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_max_grid_size=32,
    warpx_blocking_factor=4,
    warpx_potential_hi_r=VO,
    warpx_potential_lo_z=VO,
    warpx_potential_hi_z=VO,
)
rho_s = f"sqrt(x*x+y*y+(z-{ZC})**2)"
eb = picmi.EmbeddedBoundary(
    implicit_function=f"({rho_s} - a)*({rho_s} - b)",
    potential=f"if({rho_s} < 0.5*(a+b), v, vo)",
    a=A,
    b=B,
    v=V,
    vo=VO,
)
solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-12,
    warpx_absolute_tolerance=1e-9,
)

rng = np.random.default_rng(1)
theta = np.arccos(rng.uniform(-1.0, 1.0, args.np))  # polar angle from the axis
t = rng.uniform(0.0, 1.5, args.np)  # distance to the wall in cells
outer = rng.uniform(size=args.np) < 0.5
rho = np.where(outer, B - t * DX, A + t * DX)
r0 = rho * np.sin(theta)
z0 = ZC + rho * np.cos(theta)
keep = r0 > 1e-3
r0, z0 = r0[keep], z0[keep]
zeros = np.zeros_like(r0)
dist = picmi.ParticleListDistribution(
    x=r0, y=zeros, z=z0, ux=zeros, uy=zeros, uz=zeros, weight=zeros + 1e-30
)
DT = 1e-15
sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=1,
    verbose=0,
    particle_shape=args.shape,
    warpx_embedded_boundary=eb,
)
sim.add_species(
    picmi.Species(particle_type="electron", name="e", initial_distribution=dist),
    layout=picmi.PseudoRandomLayout(n_macroparticles=len(r0), grid=grid),
)
sim.initialize_inputs()

import pywarpx  # noqa: E402

pywarpx.warpx.eb_e_field = args.eb_e_field
pywarpx.warpx.eb_poisson_stencil = args.eb_poisson_stencil
sim.initialize_warpx()

from pywarpx import particle_containers  # noqa: E402

pc = particle_containers.ParticleContainerWrapper("e")


def get(name):
    return np.concatenate(pc.get_particle_real_arrays(name, 0))


u0 = {k: get(k) for k in ("ux", "uz")}
r, z = get("r"), get("z")
sim.step(1)
u1 = {k: get(k) for k in ("ux", "uz")}

# electron: du = -e E dt / m
Er = -(u1["ux"] - u0["ux"]) * m_e / (e * DT)
Ez = -(u1["uz"] - u0["uz"]) * m_e / (e * DT)
rr = np.hypot(r, z - ZC)
Er_ex, Ez_ex = K * r / rr**3, K * (z - ZC) / rr**3
err = np.hypot(Er - Er_ex, Ez - Ez_ex) / np.hypot(Er_ex, Ez_ex)
t = np.where(rr > 0.5 * (A + B), B - rr, rr - A) / DX
near = t < 0.6
out = {
    "n": args.n,
    "eb_e_field": args.eb_e_field,
    "eb_poisson_stencil": args.eb_poisson_stencil,
    "shape": args.shape,
    "particles": int(len(err)),
    "err_max_near": float(err[near].max()),
    "err_max_far": float(err[~near].max()),
}
print(json.dumps(out, indent=1))
with open("nearwall.json", "w") as f:
    json.dump(out, f, indent=1)
