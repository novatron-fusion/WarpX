
import picmistandard
import pywarpx


class LoadAppliedRFField(picmistandard.PICMI_LoadAppliedField):
    def applied_field_initialize_inputs(self):
        pywarpx.particles.read_rf_fields_from_path = self.read_fields_from_path
        if self.load_E:
            pywarpx.particles.E_ext_rf_particle_init_style = "read_from_file"
        if self.load_B:
            assert False, "Loading RF B field from file not implemented yet"


class ParticleExternalFileInjector(picmistandard.base._ClassWithInit):
    """
    Inject macroparticles with properties (mass, charge, position, and momentum - :math:`\\gamma \\beta m c`) read from an external openPMD file.
    With it users can specify the additional arguments:

    Parameters
    ----------

    injection_file: string
        openPMD file name to read the particle data from.
    charge: double
        optional (default is read from openPMD file) when set this will be the charge of
        the physical particle represented by the injected macroparticles.
    mass: double
        optional (default is read from openPMD file) when set this will be the charge of
        the physical particle represented by the injected macroparticles.
    z_shift: double
        optional (default is no shift) when set this value will be added to the longitudinal,
        z, position of the particles.
    impose_t_lab_from_file: bool
        optional (default is false) only read if warpx.gamma_boost > 1., it allows to set
        t_lab for the Lorentz Transform as being the time stored in the openPMD file.

      Warning: ``q_tot!=0`` is not supported with the ``external_file``
      injection style. If a value is provided, it is ignored and no re-scaling
      is done. The external file must include the species ``openPMD::Record``
      labeled ``position`` and ``momentum`` (`double` arrays), with
      dimensionality and units set via ``openPMD::setUnitDimension`` and
      ``setUnitSI``. If the external file also contains ``openPMD::Records`` for
      ``mass`` and ``charge`` (constant `double` scalars) then the species will
      use these, unless overwritten in the input file (see
      ``<species_name>.mass``, ``<species_name>.charge`` or
      ``<species_name>.species_type``). The ``external_file`` option is
      currently implemented for 2D, 3D and RZ geometries, with record components
      in the cartesian coordinates ``(x,y,z)`` for 3D and RZ, and ``(x,z)`` for
      2D. For more information on the `openPMD format
      <https://github.com/openPMD>`__ and how to build WarpX with it, please
      visit :ref:`the install section <install-developers>`.
    """

    def __init__(
        self,
        filename,
        charge=None,
        mass=None,
        z_shift=None,
        impose_t_lab_from_file=None,
        **kw
    ):
        self.filename = filename
        self.charge = charge
        self.mass = mass
        self.z_shift = z_shift
        self.impose_t_lab_from_file = impose_t_lab_from_file
        self.handle_init(kw)

    def distribution_initialize_inputs(
        self, species_number, layout, species, density_scale, source_name
    ):
        species.add_new_group_attr(source_name, "injection_style", "external_file")
        species.add_new_group_attr(source_name, "injection_file", self.filename)
        if hasattr(self, "charge"):
            species.add_new_group_attr(source_name, "charge", self.charge)
        if hasattr(self, "mass"):
            species.add_new_group_attr(source_name, "mass", self.mass)
        if hasattr(self, "z_shift"):
            species.add_new_group_attr(source_name, "z_shift", self.z_shift)
        if hasattr(self, "impose_t_lab_from_file"):
            species.add_new_group_attr(
                source_name, "impose_t_lab_from_file", self.impose_t_lab_from_file
            )
