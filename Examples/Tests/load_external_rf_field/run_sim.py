
from pywarpx import picmi
import pywarpx


import picmistandard

class LoadAppliedRFField(picmistandard.PICMI_LoadAppliedField):
    def applied_field_initialize_inputs(self):
        pywarpx.particles.read_rf_fields_from_path = self.read_fields_from_path
        if self.load_E:
            pywarpx.particles.E_ext_rf_particle_init_style = "read_from_file"
        if self.load_B:
            assert False, "Loading RF B field from file not implemented yet"

constants = picmi.constants

#################################
####### GENERAL PARAMETERS ######
#################################

max_steps = 200

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

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    warpx_format="openpmd",
    grid=grid,
    period=1,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", "Jx", "Jy", "Jz"],
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

sim.add_diagnostic(field_diag)

#################################
##### SIMULATION EXECUTION ######
#################################

sim.step(max_steps)

