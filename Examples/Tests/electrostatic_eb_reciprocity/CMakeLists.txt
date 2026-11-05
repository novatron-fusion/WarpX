# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_eb_reciprocity_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_electrostatic_eb_reciprocity_picmi.py --stencil symmetric"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_eb_reciprocity_cut_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_electrostatic_eb_reciprocity_picmi.py --stencil symmetric --a 0.8 1.0 --b 0.55 1.3"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_eb_reciprocity_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_electrostatic_eb_reciprocity_picmi.py --stencil symmetric"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_eb_coax_conforming_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_electrostatic_eb_coax_picmi.py --eb-e-field conforming --eb-poisson-stencil symmetric"  # inputs
        "analysis_coax.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
