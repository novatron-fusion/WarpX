#!/usr/bin/env python3
#
# Field at an embedded boundary (RZ, electrostatic): electrodes at r < r_i
# (at V) and r > r_o (at V_o, as the domain wall at r = R), cut cells, z
# periodic, no particles. Checks the vacuum potential
# V_o + (V - V_o) ln(r / r_o) / ln(r_i / r_o) and field
# E_r = (V - V_o) / (r ln(r_o / r_i)) between the electrodes, and records E_r
# on the two edges cut by the electrodes with the field at the wall
# (warpx.eb_e_field, warpx.eb_poisson_stencil). Writes coax.json.

import argparse
import json

import numpy as np

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=32, help="cells per unit length")
parser.add_argument("--eb-e-field", default="gradient")
parser.add_argument("--eb-poisson-stencil", default="shortley_weller")
args, _ = parser.parse_known_args()

R, LZ = 1.0, 2.0
NR, NZ = args.n, 2 * args.n
DR = R / NR
RI, RO = 0.2317, 0.8133
RM = 0.5 * (RI + RO)
V, VO = 50.0, -20.0

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R, LZ],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_max_grid_size=max(16, NR // 2),
    warpx_blocking_factor=4,
    warpx_potential_hi_r=VO,
)
eb = picmi.EmbeddedBoundary(
    implicit_function="(x - ri)*(x - ro)",
    potential="if(x < rm, v, vo)",
    ri=RI,
    ro=RO,
    rm=RM,
    v=V,
    vo=VO,
)
solver = picmi.ElectrostaticSolver(
    grid=grid,
    method="Multigrid",
    required_precision=1e-10,
    warpx_absolute_tolerance=1e-6,
)
sim = picmi.Simulation(
    solver=solver,
    time_step_size=1e-12,
    max_steps=1,
    verbose=0,
    particle_shape=1,
    warpx_embedded_boundary=eb,
)
sim.add_species(picmi.Species(particle_type="electron", name="electrons"), layout=None)
sim.initialize_inputs()
import pywarpx  # noqa: E402

pywarpx.warpx.eb_e_field = args.eb_e_field
pywarpx.warpx.eb_poisson_stencil = args.eb_poisson_stencil
sim.initialize_warpx()
sim.step(1)

from pywarpx import libwarpx  # noqa: E402

reg = libwarpx.libwarpx_so.get_instance().multifab_register()
mf = reg.get("phi_fp", level=0)
phi = np.asarray(mf[:, :]).reshape(mf[:, :].shape[:2])
mf = reg.get("Efield_fp", dir=libwarpx.libwarpx_so.Direction(0), level=0)
Er = np.asarray(mf[:, :]).reshape(mf[:, :].shape[:2]).mean(axis=1)

r_node = DR * np.arange(phi.shape[0])
r_cell = DR * (0.5 + np.arange(Er.shape[0]))
gap = (r_node > RI) & (r_node < RO)
phi_exact = VO + (V - VO) * np.log(r_node[gap] / RO) / np.log(RI / RO)
between = (r_cell > RI + DR) & (r_cell < RO - DR)
Er_exact = (V - VO) / (r_cell * np.log(RO / RI))
cut = [int(np.floor(RI / DR)), int(np.floor(RO / DR))]
out = {
    "n": args.n,
    "eb_e_field": args.eb_e_field,
    "eb_poisson_stencil": args.eb_poisson_stencil,
    "phi_rel_max": float(np.abs(phi[gap] - phi_exact[:, None]).max() / abs(V - VO)),
    "Er_rel_max": float(
        np.abs(Er - Er_exact)[between].max() / np.abs(Er_exact[between]).max()
    ),
    "Er_cut": [float(Er[i]) for i in cut],
    "Er_surface": [float((V - VO) / (r * np.log(RO / RI))) for r in (RI, RO)],
}
print(json.dumps(out, indent=1))
with open("coax.json", "w") as f:
    json.dump(out, f, indent=1)
