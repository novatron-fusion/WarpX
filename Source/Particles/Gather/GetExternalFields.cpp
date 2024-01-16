#include "Particles/Gather/GetExternalFields.H"

#include "AcceleratorLattice/AcceleratorLattice.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Vector.H>

#include <string>

using namespace amrex::literals;

GetExternalEBField::GetExternalEBField (const WarpXParIter& a_pti, long a_offset) noexcept
{
    auto& warpx = WarpX::GetInstance();
    auto& mypc = warpx.GetPartContainer();

    const int lev = a_pti.GetLevel();

    AcceleratorLattice const & accelerator_lattice = warpx.get_accelerator_lattice(lev);
    if (accelerator_lattice.m_lattice_defined) {
        d_lattice_element_finder = accelerator_lattice.GetFinderDeviceInstance(a_pti, static_cast<int>(a_offset));
    }

    m_gamma_boost = WarpX::gamma_boost;
    m_uz_boost = std::sqrt(WarpX::gamma_boost*WarpX::gamma_boost - 1._prt)*PhysConst::c;

    m_Etype = Unknown;
    m_Btype = Unknown;

    if (mypc.m_E_ext_particle_s == "none") { m_Etype = None; }
    if (mypc.m_B_ext_particle_s == "none") { m_Btype = None; }

    // These lines will be removed once the user interface is redefined and the CI tests updated
    if (mypc.m_E_ext_particle_s == "constant") { m_Etype = None; }
    if (mypc.m_B_ext_particle_s == "constant") { m_Btype = None; }

    if (mypc.m_E_ext_particle_s == "parse_e_ext_particle_function" ||
        mypc.m_B_ext_particle_s == "parse_b_ext_particle_function" ||
        mypc.m_E_ext_particle_s == "repeated_plasma_lens" ||
        mypc.m_B_ext_particle_s == "repeated_plasma_lens" || 
        mypc.m_E_ext_particle_s == "read_from_file" || 
        mypc.m_B_ext_particle_s == "read_from_file")
    {
        m_time = warpx.gett_new(a_pti.GetLevel());
        m_get_position = GetParticlePosition<PIdx>(a_pti, a_offset);
    }

    if (mypc.m_E_ext_particle_s == "parse_e_ext_particle_function")
    {
        m_Etype = ExternalFieldInitType::Parser;
        constexpr auto num_arguments = 4; //x,y,z,t
        m_Exfield_partparser = mypc.m_Ex_particle_parser->compile<num_arguments>();
        m_Eyfield_partparser = mypc.m_Ey_particle_parser->compile<num_arguments>();
        m_Ezfield_partparser = mypc.m_Ez_particle_parser->compile<num_arguments>();
    }

    if (mypc.m_B_ext_particle_s == "parse_b_ext_particle_function")
    {
        m_Btype = ExternalFieldInitType::Parser;
        constexpr auto num_arguments = 4; //x,y,z,t
        m_Bxfield_partparser = mypc.m_Bx_particle_parser->compile<num_arguments>();
        m_Byfield_partparser = mypc.m_By_particle_parser->compile<num_arguments>();
        m_Bzfield_partparser = mypc.m_Bz_particle_parser->compile<num_arguments>();
    }

    if (mypc.m_E_ext_particle_s == "repeated_plasma_lens" ||
        mypc.m_B_ext_particle_s == "repeated_plasma_lens")
    {
        if (mypc.m_E_ext_particle_s == "repeated_plasma_lens") { m_Etype = RepeatedPlasmaLens; }
        if (mypc.m_B_ext_particle_s == "repeated_plasma_lens") { m_Btype = RepeatedPlasmaLens; }
        m_dt = warpx.getdt(a_pti.GetLevel());
        const auto& attribs = a_pti.GetAttribs();
        m_ux = attribs[PIdx::ux].dataPtr() + a_offset;
        m_uy = attribs[PIdx::uy].dataPtr() + a_offset;
        m_uz = attribs[PIdx::uz].dataPtr() + a_offset;
        m_repeated_plasma_lens_period = mypc.m_repeated_plasma_lens_period;
        m_n_lenses = static_cast<int>(mypc.h_repeated_plasma_lens_starts.size());
        m_repeated_plasma_lens_starts = mypc.d_repeated_plasma_lens_starts.data();
        m_repeated_plasma_lens_lengths = mypc.d_repeated_plasma_lens_lengths.data();
        m_repeated_plasma_lens_strengths_E = mypc.d_repeated_plasma_lens_strengths_E.data();
        m_repeated_plasma_lens_strengths_B = mypc.d_repeated_plasma_lens_strengths_B.data();
    }

#if defined(WARPX_DIM_3D)

    if (mypc.m_E_ext_particle_s == "read_from_file" || 
        mypc.m_B_ext_particle_s == "read_from_file") 
    {
        if (mypc.m_E_ext_particle_s == "read_from_file") m_Etype = ReadFromFile;
        if (mypc.m_B_ext_particle_s == "read_from_file") m_Btype = ReadFromFile;

        auto series = openPMD::Series(mypc.m_read_fields_from_path, openPMD::Access::READ_ONLY);
        auto iseries = series.iterations.begin()->second;
        auto F = iseries.meshes["B"];

        auto axisLabels = F.getAttribute("axisLabels").get<std::vector<std::string>>();
        auto fileGeom = F.getAttribute("geometry").get<std::string>();

        if (fileGeom == "cartesian")
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "x" && axisLabels[1] == "y" && axisLabels[2] == "z",
                                            "3D expects axisLabels {x, y, z}");

            const auto offset = F.gridGlobalOffset();
            const auto gridSpacing = F.gridSpacing<long double>();
            auto FCx = F["x"];
            auto FCy = F["y"];
            auto FCz = F["z"];
            const auto extent = FCx.getExtent();

            const openPMD::Offset chunk_offset = {0,0,0};
            const openPMD::Extent chunk_extent = {extent[0], extent[1], extent[2]};

            auto FCx_chunk_data = FCx.loadChunk<amrex::Real>(chunk_offset,chunk_extent);
            auto FCy_chunk_data = FCy.loadChunk<amrex::Real>(chunk_offset,chunk_extent);
            auto FCz_chunk_data = FCz.loadChunk<amrex::Real>(chunk_offset,chunk_extent);
            series.flush();
            auto FCx_data_host = FCx_chunk_data.get();
            auto FCy_data_host = FCy_chunk_data.get();
            auto FCz_data_host = FCz_chunk_data.get();
            const size_t total_extent = size_t(extent[0]) * extent[1] * extent[2];
            amrex::Gpu::DeviceVector<amrex::Real> FCx_data_gpu(total_extent);
            amrex::Gpu::DeviceVector<amrex::Real> FCy_data_gpu(total_extent);
            amrex::Gpu::DeviceVector<amrex::Real> FCz_data_gpu(total_extent);

            auto FCx_data = FCx_data_gpu.data();
            auto FCy_data = FCy_data_gpu.data();
            auto FCz_data = FCz_data_gpu.data();

            amrex::Gpu::copy(amrex::Gpu::hostToDevice, FCx_data_host, FCx_data_host + total_extent, FCx_data);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, FCy_data_host, FCy_data_host + total_extent, FCy_data);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, FCz_data_host, FCz_data_host + total_extent, FCz_data);

            const amrex::Array4<amrex::Real> fcx_array(FCx_data, {0,0,0}, {extent[2], extent[1], extent[0]}, 1);
            const amrex::Array4<amrex::Real> fcy_array(FCy_data, {0,0,0}, {extent[2], extent[1], extent[0]}, 1);
            const amrex::Array4<amrex::Real> fcz_array(FCz_data, {0,0,0}, {extent[2], extent[1], extent[0]}, 1);

            Bfield_file_external_particle_cart = new ExternalFieldFromFile3DCart(
                amrex::RealVect {gridSpacing[0], gridSpacing[1], gridSpacing[2] }, 
                amrex::RealVect {offset[0], offset[1], offset[2] }, 
                fcx_array, fcy_array, fcz_array
            );
        } else if (fileGeom == "thetaMode")
        {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "r" && axisLabels[1] == "z",
                                            "RZ expects axisLabels {r, z}");

            const auto offset = F.gridGlobalOffset();
            const auto gridSpacing = F.gridSpacing<long double>();
            auto FCr = F["r"];
            auto FCz = F["z"];
            const auto extent = FCr.getExtent();

            const openPMD::Offset chunk_offset = {0,0,0};
            const openPMD::Extent chunk_extent = {extent[0], extent[1], extent[2]};

            auto FCr_chunk_data = FCr.loadChunk<amrex::Real>(chunk_offset,chunk_extent);
            auto FCz_chunk_data = FCz.loadChunk<amrex::Real>(chunk_offset,chunk_extent);
            series.flush();
            auto FCr_data_host = FCr_chunk_data.get();
            auto FCz_data_host = FCz_chunk_data.get();
            const size_t total_extent = size_t(extent[0]) * extent[1] * extent[2];
            amrex::Gpu::DeviceVector<amrex::Real> FCr_data_gpu(total_extent);
            amrex::Gpu::DeviceVector<amrex::Real> FCz_data_gpu(total_extent);

            auto FCr_data = FCr_data_gpu.data();
            auto FCz_data = FCz_data_gpu.data();

            amrex::Gpu::copy(amrex::Gpu::hostToDevice, FCr_data_host, FCr_data_host + total_extent, FCr_data);
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, FCz_data_host, FCz_data_host + total_extent, FCz_data);

            const amrex::Array4<amrex::Real> fcr_array(FCr_data, {0,0,0}, {extent[0], extent[2], extent[1]}, 1);
            const amrex::Array4<amrex::Real> fcz_array(FCz_data, {0,0,0}, {extent[0], extent[2], extent[1]}, 1);

            Bfield_file_external_particle_cyl = new ExternalFieldFromFile3DCyl(
                amrex::RealVect {gridSpacing[0], gridSpacing[1] }, 
                amrex::RealVect {offset[0], offset[1], offset[2] }, 
                fcr_array, fcz_array
            );
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false, "3D can only read from files with cartesian or thetaMode geometry");
        }
    }
#endif

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Etype != Unknown, "Unknown E_ext_particle_init_style");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_Btype != Unknown, "Unknown B_ext_particle_init_style");

}
