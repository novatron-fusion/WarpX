
from pywarpx import picmi

from picmi_ext import ParticleExternalFileInjector, LoadAppliedRFField

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 150000

nx = 256
ny = 256
nz = 384
max_grid_size = max(nx, ny, nz)

xmin = -0.75
xmax = 0.75
ymin = xmin
ymax = xmax
zmin = -1.125
zmax = 1.125

#################################
############ NUMERICS ###########
#################################

verbose = 1
use_filter = 0

# Order of particle shape factors
particle_shape = 3


#################################
############ PLASMA #############
#################################

proton_dist = ParticleExternalFileInjector(
    filename="protons.h5"
)

protons = picmi.Species(
    particle_type="H",
    name="protons",
    charge="q_e",
    mass="m_p",
    initial_distribution=proton_dist,
)

electron_dist = ParticleExternalFileInjector(
    filename="electrons.h5"
)

electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
    charge="-q_e",
    mass="m_e",
    initial_distribution=electron_dist,
)


#################################
######## INITIAL FIELD ##########
#################################

B_ext = picmi.LoadAppliedField(
    read_fields_from_path="B.h5",
    load_E=False,
)

RF_ext = LoadAppliedRFField(
    read_fields_from_path="RF.h5",
    load_E=True,
    load_B=False,
)

#################################
###### GRID AND SOLVER ##########
#################################


grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    warpx_max_grid_size=max_grid_size,
    lower_bound=[xmin, ymin, zmin],
    upper_bound=[xmax, ymax, zmax],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
    lower_boundary_conditions_particles=["open", "open", "open"],
    upper_boundary_conditions_particles=["open", "open", "open"],
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9, divE_cleaning=0)

#################################
######### DIAGNOSTICS ###########
#################################

particle_diag = picmi.ParticleDiagnostic(
    name="diag1",
    warpx_format="openpmd",
    period=500,
    species=[protons, electrons],
    data_list=["ux", "uy", "uz", "x", "y", "z", "weighting"],
)
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    warpx_format="openpmd",
    grid=grid,
    period=500,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz", "rho_electrons", "rho_protons"],
)

#################################
####### SIMULATION SETUP ########
#################################

sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=verbose,
    warpx_serialize_initial_conditions=False,
    warpx_do_dynamic_scheduling=False,
    warpx_use_filter=use_filter,
    particle_shape=particle_shape,
    warpx_amrex_the_arena_init_size=23*1024*1024*1024,
    warpx_amrex_use_gpu_aware_mpi=True,
)

sim.add_applied_field(B_ext)
sim.add_applied_field(RF_ext)


sim.add_species(protons, layout=None)
 
sim.add_species(electrons, layout=None)


sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_diag)

#################################
##### SIMULATION EXECUTION ######
#################################

sim.step(max_steps)

