/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "HybridPICModel.H"

#include "FieldSolver/Fields.H"
#include "WarpX.H"

#include "Utils/Algorithms/LinearInterpolation.H"

using namespace amrex;
using namespace warpx::fields;

HybridPICModel::HybridPICModel ( int nlevs_max )
{
    ReadParameters();
    AllocateMFs(nlevs_max);
}

void HybridPICModel::ReadParameters ()
{
    const ParmParse pp_hybrid("hybrid_pic_model");

    // The B-field update is subcycled to improve stability - the number
    // of sub steps can be specified by the user (defaults to 50).
    utils::parser::queryWithParser(pp_hybrid, "substeps", m_substeps);

    // The hybrid model requires an electron temperature, reference density
    // and exponent to be given. These values will be used to calculate the
    // electron pressure according to p = n0 * Te * (n/n0)^gamma
    utils::parser::queryWithParser(pp_hybrid, "gamma", m_gamma);
    if (!utils::parser::queryWithParser(pp_hybrid, "elec_temp", m_elec_temp)) {
        Abort("hybrid_pic_model.elec_temp must be specified when using the hybrid solver");
    }
    const bool n0_ref_given = utils::parser::queryWithParser(pp_hybrid, "n0_ref", m_n0_ref);
    if (m_gamma != 1.0 && !n0_ref_given) {
        Abort("hybrid_pic_model.n0_ref should be specified if hybrid_pic_model.gamma != 1");
    }

    pp_hybrid.query("plasma_resistivity(rho,J)", m_eta_expression);
    utils::parser::queryWithParser(pp_hybrid, "n_floor", m_n_floor);

    utils::parser::queryWithParser(pp_hybrid, "plasma_hyper_resistivity", m_eta_h);

    // convert electron temperature from eV to J
    m_elec_temp *= PhysConst::q_e;

    // external currents
    pp_hybrid.query("Jx_external_grid_function(x,y,z,t)", m_Jx_ext_grid_function);
    pp_hybrid.query("Jy_external_grid_function(x,y,z,t)", m_Jy_ext_grid_function);
    pp_hybrid.query("Jz_external_grid_function(x,y,z,t)", m_Jz_ext_grid_function);

    // external magnetic field
    pp_hybrid.query("Bx_external_grid_function(x,y,z)", m_Bx_ext_grid_function);
    pp_hybrid.query("By_external_grid_function(x,y,z)", m_By_ext_grid_function);
    pp_hybrid.query("Bz_external_grid_function(x,y,z)", m_Bz_ext_grid_function);

    pp_hybrid.query("B_external_init_style", B_ext_grid_s);

    amrex::Print() << B_ext_grid_s << "\n";
    if (B_ext_grid_s == "read_from_file"){
            const std::string read_fields_from_path="./";
            pp_hybrid.query("read_fields_from_path", external_fields_path);
            amrex::Print() << external_fields_path << "\n";
    }
}

void HybridPICModel::AllocateMFs (int nlevs_max)
{
    electron_pressure_fp.resize(nlevs_max);
    rho_fp_temp.resize(nlevs_max);
    current_fp_temp.resize(nlevs_max);
    current_fp_ampere.resize(nlevs_max);
    current_fp_external.resize(nlevs_max);
    bfield_fp_external.resize(nlevs_max);
}

void HybridPICModel::AllocateLevelMFs (int lev, const BoxArray& ba, const DistributionMapping& dm,
                                       const int ncomps, const IntVect& ngJ, const IntVect& ngRho,
                                       const IntVect& jx_nodal_flag,
                                       const IntVect& jy_nodal_flag,
                                       const IntVect& jz_nodal_flag,
                                       const IntVect& rho_nodal_flag)
{
    // The "electron_pressure_fp" multifab stores the electron pressure calculated
    // from the specified equation of state.
    // The "rho_fp_temp" multifab is used to store the ion charge density
    // interpolated or extrapolated to appropriate timesteps.
    // The "current_fp_temp" multifab is used to store the ion current density
    // interpolated or extrapolated to appropriate timesteps.
    // The "current_fp_ampere" multifab stores the total current calculated as
    // the curl of B.
    WarpX::AllocInitMultiFab(electron_pressure_fp[lev], amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, lev, "electron_pressure_fp", 0.0_rt);

    WarpX::AllocInitMultiFab(rho_fp_temp[lev], amrex::convert(ba, rho_nodal_flag),
        dm, ncomps, ngRho, lev, "rho_fp_temp", 0.0_rt);

    WarpX::AllocInitMultiFab(current_fp_temp[lev][0], amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_temp[x]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_temp[lev][1], amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_temp[y]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_temp[lev][2], amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_temp[z]", 0.0_rt);

    WarpX::AllocInitMultiFab(current_fp_ampere[lev][0], amrex::convert(ba, jx_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_ampere[x]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_ampere[lev][1], amrex::convert(ba, jy_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_ampere[y]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_ampere[lev][2], amrex::convert(ba, jz_nodal_flag),
        dm, ncomps, ngJ, lev, "current_fp_ampere[z]", 0.0_rt);

    // the external current density multifab is made nodal to avoid needing to interpolate
    // to a nodal grid as has to be done for the ion and total current density multifabs
    // this also allows the external current multifab to not have any ghost cells
    WarpX::AllocInitMultiFab(current_fp_external[lev][0], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "current_fp_external[x]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_external[lev][1], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "current_fp_external[y]", 0.0_rt);
    WarpX::AllocInitMultiFab(current_fp_external[lev][2], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "current_fp_external[z]", 0.0_rt);

    // external magnetic field
    WarpX::AllocInitMultiFab(bfield_fp_external[lev][0], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "bfield_fp_external[x]", 0.0_rt);
    WarpX::AllocInitMultiFab(bfield_fp_external[lev][1], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "bfield_fp_external[y]", 0.0_rt);
    WarpX::AllocInitMultiFab(bfield_fp_external[lev][2], amrex::convert(ba, IntVect(AMREX_D_DECL(1,1,1))),
        dm, ncomps, IntVect(AMREX_D_DECL(0,0,0)), lev, "bfield_fp_external[z]", 0.0_rt);

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (ncomps == 1),
        "Ohm's law solver only support m = 0 azimuthal mode at present.");
#endif
}

void HybridPICModel::ClearLevel (int lev)
{
    electron_pressure_fp[lev].reset();
    rho_fp_temp[lev].reset();
    for (int i = 0; i < 3; ++i) {
        current_fp_temp[lev][i].reset();
        current_fp_ampere[lev][i].reset();
        current_fp_external[lev][i].reset();
    }
}

void HybridPICModel::InitData ()
{
    m_resistivity_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_eta_expression, {"rho","J"}));
    m_eta = m_resistivity_parser->compile<2>();
    const std::set<std::string> resistivity_symbols = m_resistivity_parser->symbols();
    m_resistivity_has_J_dependence += resistivity_symbols.count("J");

    m_J_external_parser[0] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_Jx_ext_grid_function,{"x","y","z","t"}));
    m_J_external_parser[1] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_Jy_ext_grid_function,{"x","y","z","t"}));
    m_J_external_parser[2] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_Jz_ext_grid_function,{"x","y","z","t"}));
    m_J_external[0] = m_J_external_parser[0]->compile<4>();
    m_J_external[1] = m_J_external_parser[1]->compile<4>();
    m_J_external[2] = m_J_external_parser[2]->compile<4>();

    // check if the external current parsers depend on time
    for (int i=0; i<3; i++) {
        const std::set<std::string> J_ext_symbols = m_J_external_parser[i]->symbols();
        m_external_field_has_time_dependence += J_ext_symbols.count("t");
    }

    m_B_external_parser[0] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_Bx_ext_grid_function,{"x","y","z"}));
    m_B_external_parser[1] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_By_ext_grid_function,{"x","y","z"}));
    m_B_external_parser[2] = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(m_Bz_ext_grid_function,{"x","y","z"}));
    m_B_external[0] = m_B_external_parser[0]->compile<3>();
    m_B_external[1] = m_B_external_parser[1]->compile<3>();
    m_B_external[2] = m_B_external_parser[2]->compile<3>();

    auto & warpx = WarpX::GetInstance();

    // Get the grid staggering of the fields involved in calculating E
    amrex::IntVect Jx_stag = warpx.getField(FieldType::current_fp, 0,0).ixType().toIntVect();
    amrex::IntVect Jy_stag = warpx.getField(FieldType::current_fp, 0,1).ixType().toIntVect();
    amrex::IntVect Jz_stag = warpx.getField(FieldType::current_fp, 0,2).ixType().toIntVect();
    amrex::IntVect Bx_stag = warpx.getField(FieldType::Bfield_fp, 0,0).ixType().toIntVect();
    amrex::IntVect By_stag = warpx.getField(FieldType::Bfield_fp, 0,1).ixType().toIntVect();
    amrex::IntVect Bz_stag = warpx.getField(FieldType::Bfield_fp, 0,2).ixType().toIntVect();
    amrex::IntVect Ex_stag = warpx.getField(FieldType::Efield_fp, 0,0).ixType().toIntVect();
    amrex::IntVect Ey_stag = warpx.getField(FieldType::Efield_fp, 0,1).ixType().toIntVect();
    amrex::IntVect Ez_stag = warpx.getField(FieldType::Efield_fp, 0,2).ixType().toIntVect();

    // Check that the grid types are appropriate
    const bool appropriate_grids = (
#if   defined(WARPX_DIM_1D_Z)
        // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
        Ex_stag == IntVect(1) && Ey_stag == IntVect(1) && Ez_stag == IntVect(0) &&
        Bx_stag == IntVect(0) && By_stag == IntVect(0) && Bz_stag == IntVect(1) &&
#elif   defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
        Ex_stag == IntVect(0,1) && Ey_stag == IntVect(1,1) && Ez_stag == IntVect(1,0) &&
        Bx_stag == IntVect(1,0) && By_stag == IntVect(0,0) && Bz_stag == IntVect(0,1) &&
#elif defined(WARPX_DIM_3D)
        Ex_stag == IntVect(0,1,1) && Ey_stag == IntVect(1,0,1) && Ez_stag == IntVect(1,1,0) &&
        Bx_stag == IntVect(1,0,0) && By_stag == IntVect(0,1,0) && Bz_stag == IntVect(0,0,1) &&
#endif
        Jx_stag == Ex_stag && Jy_stag == Ey_stag && Jz_stag == Ez_stag
    );
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        appropriate_grids,
        "Ohm's law E-solve only works with staggered (Yee) grids.");

    // copy data to device
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        Jx_IndexType[idim]    = Jx_stag[idim];
        Jy_IndexType[idim]    = Jy_stag[idim];
        Jz_IndexType[idim]    = Jz_stag[idim];
        Bx_IndexType[idim]    = Bx_stag[idim];
        By_IndexType[idim]    = By_stag[idim];
        Bz_IndexType[idim]    = Bz_stag[idim];
        Ex_IndexType[idim]    = Ex_stag[idim];
        Ey_IndexType[idim]    = Ey_stag[idim];
        Ez_IndexType[idim]    = Ez_stag[idim];
    }

    // Below we set all the unused dimensions to have nodal values for J, B & E
    // since these values will be interpolated onto a nodal grid - if this is
    // not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z)
    Jx_IndexType[2]    = 1;
    Jy_IndexType[2]    = 1;
    Jz_IndexType[2]    = 1;
    Bx_IndexType[2]    = 1;
    By_IndexType[2]    = 1;
    Bz_IndexType[2]    = 1;
    Ex_IndexType[2]    = 1;
    Ey_IndexType[2]    = 1;
    Ez_IndexType[2]    = 1;
#endif
#if defined(WARPX_DIM_1D_Z)
    Jx_IndexType[1]    = 1;
    Jy_IndexType[1]    = 1;
    Jz_IndexType[1]    = 1;
    Bx_IndexType[1]    = 1;
    By_IndexType[1]    = 1;
    Bz_IndexType[1]    = 1;
    Ex_IndexType[1]    = 1;
    Ey_IndexType[1]    = 1;
    Ez_IndexType[1]    = 1;
#endif

    // Initialize external current - note that this approach skips the check
    // if the current is time dependent which is what needs to be done to
    // write time independent fields on the first step.
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
#ifdef AMREX_USE_EB
        auto& edge_lengths_x = warpx.getField(FieldType::edge_lengths, lev, 0);
        auto& edge_lengths_y = warpx.getField(FieldType::edge_lengths, lev, 1);
        auto& edge_lengths_z = warpx.getField(FieldType::edge_lengths, lev, 2);

        const auto edge_lengths = std::array< std::unique_ptr<amrex::MultiFab>, 3 >{
            std::make_unique<amrex::MultiFab>(
                edge_lengths_x, amrex::make_alias, 0, edge_lengths_x.nComp()),
            std::make_unique<amrex::MultiFab>(
                edge_lengths_y, amrex::make_alias, 0, edge_lengths_y.nComp()),
            std::make_unique<amrex::MultiFab>(
                edge_lengths_z, amrex::make_alias, 0, edge_lengths_z.nComp())
        };
#else
        const auto edge_lengths = std::array< std::unique_ptr<amrex::MultiFab>, 3 >();
#endif
        GetCurrentExternal(edge_lengths, lev);

        if (B_ext_grid_s == "parse_b_ext_grid_function") {
            amrex::Print() << "parse_b_ext_grid_function" << "\n";
            GetExternalBField(edge_lengths, lev);
        }
        
        if (B_ext_grid_s == "read_from_file") {
            amrex::Print() << "read_from_file" << "\n";
            ReadExternalBFieldFromFile(external_fields_path, edge_lengths, lev, bfield_fp_external[lev][0].get(), "B", "x");
            ReadExternalBFieldFromFile(external_fields_path, edge_lengths, lev, bfield_fp_external[lev][1].get(), "B", "y");
            ReadExternalBFieldFromFile(external_fields_path, edge_lengths, lev, bfield_fp_external[lev][2].get(), "B", "z");
            amrex::Print() << "Done" << "\n";
        }
    }
}

void HybridPICModel::GetExternalBField (
    const amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& edge_lengths)
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        GetExternalBField(edge_lengths[lev], lev);
    }
}


void HybridPICModel::GetExternalBField (
    const std::array< std::unique_ptr<amrex::MultiFab>, 3>& edge_lengths,
    int lev)
{
    // This logic matches closely to WarpX::InitializeExternalFieldsOnGridUsingParser
    // except that the parsers include time dependence.
    auto & warpx = WarpX::GetInstance();

    auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();

    auto& mfx = bfield_fp_external[lev][0];
    auto& mfy = bfield_fp_external[lev][1];
    auto& mfz = bfield_fp_external[lev][2];

    const amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    const amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    const amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();

    // avoid implicit lambda capture
    auto Bx_external = m_B_external[0];
    auto By_external = m_B_external[1];
    auto Bz_external = m_B_external[2];

    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
       const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, mfx->nGrowVect() );
       const amrex::Box& tby = mfi.tilebox( y_nodal_flag, mfy->nGrowVect() );
       const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, mfz->nGrowVect() );

       auto const& mfxfab = mfx->array(mfi);
       auto const& mfyfab = mfy->array(mfi);
       auto const& mfzfab = mfz->array(mfi);

#ifdef AMREX_USE_EB
       amrex::Array4<amrex::Real> const& lx = edge_lengths[0]->array(mfi);
       amrex::Array4<amrex::Real> const& ly = edge_lengths[1]->array(mfi);
       amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::ignore_unused(ly);
#endif
#else
       amrex::ignore_unused(edge_lengths);
#endif

        amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (lx(i, j, k) <= 0) return;
#endif
                // Shift required in the x-, y-, or z- position
                // depending on the index type of the multifab
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#else
                const amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the x-component of the field.
                mfxfab(i,j,k) = Bx_external(x,y,z);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (ly(i, j, k) <= 0) return;
#endif
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif defined(WARPX_DIM_3D)
                const amrex::Real fac_x = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the y-component of the field.
                mfyfab(i,j,k) = By_external(x,y,z);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (lz(i, j, k) <= 0) return;
#endif
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif defined(WARPX_DIM_3D)
                const amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the z-component of the field.
                mfzfab(i,j,k) = Bz_external(x,y,z);
            }
        );
    }
}

void HybridPICModel::GetCurrentExternal (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths)
{
    if (!m_external_field_has_time_dependence) { return; }

    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        GetCurrentExternal(edge_lengths[lev], lev);
    }
}


void HybridPICModel::GetCurrentExternal (
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& edge_lengths,
    int lev)
{
    // This logic matches closely to WarpX::InitializeExternalFieldsOnGridUsingParser
    // except that the parsers include time dependence.
    auto & warpx = WarpX::GetInstance();

    auto t = warpx.gett_new(lev);

    auto dx_lev = warpx.Geom(lev).CellSizeArray();
    const RealBox& real_box = warpx.Geom(lev).ProbDomain();

    auto& mfx = current_fp_external[lev][0];
    auto& mfy = current_fp_external[lev][1];
    auto& mfz = current_fp_external[lev][2];

    const amrex::IntVect x_nodal_flag = mfx->ixType().toIntVect();
    const amrex::IntVect y_nodal_flag = mfy->ixType().toIntVect();
    const amrex::IntVect z_nodal_flag = mfz->ixType().toIntVect();

    // avoid implicit lambda capture
    auto Jx_external = m_J_external[0];
    auto Jy_external = m_J_external[1];
    auto Jz_external = m_J_external[2];

    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
       const amrex::Box& tbx = mfi.tilebox( x_nodal_flag, mfx->nGrowVect() );
       const amrex::Box& tby = mfi.tilebox( y_nodal_flag, mfy->nGrowVect() );
       const amrex::Box& tbz = mfi.tilebox( z_nodal_flag, mfz->nGrowVect() );

       auto const& mfxfab = mfx->array(mfi);
       auto const& mfyfab = mfy->array(mfi);
       auto const& mfzfab = mfz->array(mfi);

#ifdef AMREX_USE_EB
       amrex::Array4<amrex::Real> const& lx = edge_lengths[0]->array(mfi);
       amrex::Array4<amrex::Real> const& ly = edge_lengths[1]->array(mfi);
       amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::ignore_unused(ly);
#endif
#else
       amrex::ignore_unused(edge_lengths);
#endif

        amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (lx(i, j, k) <= 0) return;
#endif
                // Shift required in the x-, y-, or z- position
                // depending on the index type of the multifab
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#else
                const amrex::Real fac_x = (1._rt - x_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - x_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - x_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the x-component of the field.
                mfxfab(i,j,k) = Jx_external(x,y,z,t);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (ly(i, j, k) <= 0) return;
#endif
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif defined(WARPX_DIM_3D)
                const amrex::Real fac_x = (1._rt - y_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - y_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - y_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the y-component of the field.
                mfyfab(i,j,k) = Jy_external(x,y,z,t);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if (lz(i, j, k) <= 0) return;
#endif
#if defined(WARPX_DIM_1D_Z)
                const amrex::Real x = 0._rt;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real z = j*dx_lev[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real y = 0._rt;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real z = j*dx_lev[1] + real_box.lo(1) + fac_z;
#elif defined(WARPX_DIM_3D)
                const amrex::Real fac_x = (1._rt - z_nodal_flag[0]) * dx_lev[0] * 0.5_rt;
                const amrex::Real x = i*dx_lev[0] + real_box.lo(0) + fac_x;
                const amrex::Real fac_y = (1._rt - z_nodal_flag[1]) * dx_lev[1] * 0.5_rt;
                const amrex::Real y = j*dx_lev[1] + real_box.lo(1) + fac_y;
                const amrex::Real fac_z = (1._rt - z_nodal_flag[2]) * dx_lev[2] * 0.5_rt;
                const amrex::Real z = k*dx_lev[2] + real_box.lo(2) + fac_z;
#endif
                // Initialize the z-component of the field.
                mfzfab(i,j,k) = Jz_external(x,y,z,t);
            }
        );
    }
}

#if defined(WARPX_USE_OPENPMD) && !defined(WARPX_DIM_1D_Z) && !defined(WARPX_DIM_XZ)
void
HybridPICModel::ReadExternalBFieldFromFile (
       const std::string& read_fields_from_path, 
       const std::array< std::unique_ptr<amrex::MultiFab>, 3>& edge_lengths,
       int lev, 
       amrex::MultiFab* mf,
       const std::string& F_name, 
       const std::string& F_component)
{
    // Get WarpX domain info
    auto& warpx = WarpX::GetInstance();
    amrex::Geometry const& geom = warpx.Geom(lev);
    const amrex::RealBox& real_box = geom.ProbDomain();
    const auto dx = geom.CellSizeArray();
    const amrex::IntVect nodal_flag = mf->ixType().toIntVect();

    // Read external field openPMD data
    auto series = openPMD::Series(read_fields_from_path, openPMD::Access::READ_ONLY);
    auto iseries = series.iterations.begin()->second;
    auto F = iseries.meshes[F_name];

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(F.getAttribute("dataOrder").get<std::string>() == "C",
                                     "Reading from files with non-C dataOrder is not implemented");

    auto axisLabels = F.getAttribute("axisLabels").get<std::vector<std::string>>();
    auto fileGeom = F.getAttribute("geometry").get<std::string>();

#if defined(WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "3D can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "x" && axisLabels[1] == "y" && axisLabels[2] == "z",
                                     "3D expects axisLabels {x, y, z}");
#elif defined(WARPX_DIM_XZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "XZ can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "x" && axisLabels[1] == "z",
                                     "XZ expects axisLabels {x, z}");
#elif defined(WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE(
        "Reading from openPMD for external fields is not known to work with 1D3V (see #3830)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "cartesian", "1D3V can only read from files with cartesian geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "z");
#elif defined(WARPX_DIM_RZ)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fileGeom == "thetaMode", "RZ can only read from files with 'thetaMode'  geometry");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(axisLabels[0] == "r" && axisLabels[1] == "z",
                                     "RZ expects axisLabels {r, z}");
#endif

    const auto offset = F.gridGlobalOffset();
    const auto offset0 = static_cast<amrex::Real>(offset[0]);
    const auto offset1 = static_cast<amrex::Real>(offset[1]);
#if defined(WARPX_DIM_3D)
    const auto offset2 = static_cast<amrex::Real>(offset[2]);
#endif
    const auto d = F.gridSpacing<long double>();

#if defined(WARPX_DIM_RZ)
    const auto file_dr = static_cast<amrex::Real>(d[0]);
    const auto file_dz = static_cast<amrex::Real>(d[1]);
#elif defined(WARPX_DIM_3D)
    const auto file_dx = static_cast<amrex::Real>(d[0]);
    const auto file_dy = static_cast<amrex::Real>(d[1]);
    const auto file_dz = static_cast<amrex::Real>(d[2]);
#endif

    auto FC = F[F_component];
    const auto extent = FC.getExtent();
    const auto extent0 = static_cast<int>(extent[0]);
    const auto extent1 = static_cast<int>(extent[1]);
    const auto extent2 = static_cast<int>(extent[2]);

    // Determine the chunk data that will be loaded.
    // Now, the full range of data is loaded.
    // Loading chunk data can speed up the process.
    // Thus, `chunk_offset` and `chunk_extent` should be modified accordingly in another PR.
    const openPMD::Offset chunk_offset = {0,0,0};
    const openPMD::Extent chunk_extent = {extent[0], extent[1], extent[2]};

    auto FC_chunk_data = FC.loadChunk<double>(chunk_offset,chunk_extent);
    series.flush();
    auto *FC_data_host = FC_chunk_data.get();

    // Load data to GPU
    const size_t total_extent = size_t(extent[0]) * extent[1] * extent[2];
    amrex::Gpu::DeviceVector<double> FC_data_gpu(total_extent);
    auto *FC_data = FC_data_gpu.data();
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, FC_data_host, FC_data_host + total_extent, FC_data);

    // Loop over boxes
    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box box = mfi.growntilebox();
        const amrex::Box tb = mfi.tilebox(nodal_flag, mf->nGrowVect());
        auto const& mffab = mf->array(mfi);

#ifdef AMREX_USE_EB
        amrex::Array4<amrex::Real> const& lx = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const& ly = edge_lengths[1]->array(mfi);
        amrex::Array4<amrex::Real> const& lz = edge_lengths[2]->array(mfi);
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::ignore_unused(ly);
#endif
#else
        amrex::ignore_unused(edge_lengths);
#endif

        // Start ParallelFor
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // skip if node is covered by an embedded boundary
#ifdef AMREX_USE_EB
                if ((lx(i, j, k) <= 0) || (ly(i, j, k) <= 0) || (lz(i, j, k) <= 0)) return;
#endif
                // i,j,k denote x,y,z indices in 3D xyz.
                // i,j denote r,z indices in 2D rz; k is just 0

                // ii is used for 2D RZ mode
#if defined(WARPX_DIM_RZ)
                // In 2D RZ, i denoting r can be < 0
                // but mirrored values should be assigned.
                // Namely, mffab(i) = FC_data[-i] when i<0.
                const int ii = (i<0)?(-i):(i);
#else
                const int ii = i;
#endif

                // Physical coordinates of the grid point
                // 0,1,2 denote x,y,z in 3D xyz.
                // 0,1 denote r,z in 2D rz.
                amrex::Real x0, x1;
                if ( box.type(0)==amrex::IndexType::CellIndex::NODE )
                     { x0 = static_cast<amrex::Real>(real_box.lo(0)) + ii*dx[0]; }
                else { x0 = static_cast<amrex::Real>(real_box.lo(0)) + ii*dx[0] + 0.5_rt*dx[0]; }
                if ( box.type(1)==amrex::IndexType::CellIndex::NODE )
                     { x1 = real_box.lo(1) + j*dx[1]; }
                else { x1 = real_box.lo(1) + j*dx[1] + 0.5_rt*dx[1]; }

#if defined(WARPX_DIM_RZ)
                // Get index of the external field array
                int const ir = std::floor( (x0-offset0)/file_dr );
                int const iz = std::floor( (x1-offset1)/file_dz );

                // Get coordinates of external grid point
                amrex::Real const xx0 = offset0 + ir * file_dr;
                amrex::Real const xx1 = offset1 + iz * file_dz;

#elif defined(WARPX_DIM_3D)
                amrex::Real x2;
                if ( box.type(2)==amrex::IndexType::CellIndex::NODE )
                     { x2 = real_box.lo(2) + k*dx[2]; }
                else { x2 = real_box.lo(2) + k*dx[2] + 0.5_rt*dx[2]; }

                // Get index of the external field array
                int const ix = std::floor( (x0-offset0)/file_dx );
                int const iy = std::floor( (x1-offset1)/file_dy );
                int const iz = std::floor( (x2-offset2)/file_dz );

                // Get coordinates of external grid point
                amrex::Real const xx0 = offset0 + ix * file_dx;
                amrex::Real const xx1 = offset1 + iy * file_dy;
                amrex::Real const xx2 = offset2 + iz * file_dz;
#endif

#if defined(WARPX_DIM_RZ)
                const amrex::Array4<double> fc_array(FC_data, {0,0,0}, {extent0, extent2, extent1}, 1);
                const double
                    f00 = fc_array(0, iz  , ir  ),
                    f01 = fc_array(0, iz  , ir+1),
                    f10 = fc_array(0, iz+1, ir  ),
                    f11 = fc_array(0, iz+1, ir+1);
                mffab(i,j,k) = static_cast<amrex::Real>(utils::algorithms::bilinear_interp<double>
                    (xx0, xx0+file_dr, xx1, xx1+file_dz,
                     f00, f01, f10, f11,
                     x0, x1));
#elif defined(WARPX_DIM_3D)
                const amrex::Array4<double> fc_array(FC_data, {0,0,0}, {extent2, extent1, extent0}, 1);
                const double
                    f000 = fc_array(iz  , iy  , ix  ),
                    f001 = fc_array(iz+1, iy  , ix  ),
                    f010 = fc_array(iz  , iy+1, ix  ),
                    f011 = fc_array(iz+1, iy+1, ix  ),
                    f100 = fc_array(iz  , iy  , ix+1),
                    f101 = fc_array(iz+1, iy  , ix+1),
                    f110 = fc_array(iz  , iy+1, ix+1),
                    f111 = fc_array(iz+1, iy+1, ix+1);
                mffab(i,j,k) = static_cast<amrex::Real>(utils::algorithms::trilinear_interp<double>
                    (xx0, xx0+file_dx, xx1, xx1+file_dy, xx2, xx2+file_dz,
                     f000, f001, f010, f011, f100, f101, f110, f111,
                     x0, x1, x2));
#endif
            }

        ); // End ParallelFor

    } // End loop over boxes.

} // End function HybridPICModel::ReadCurrentExternalFromFile
#else // WARPX_USE_OPENPMD && !WARPX_DIM_1D_Z && !defined(WARPX_DIM_XZ)
void
HybridPICModel::ReadExternalBFieldFromFile (
    const std::string& , 
    const std::array< std::unique_ptr<amrex::MultiFab>, 3>& , 
    int , 
    amrex::MultiFab* , 
    const std::string& , 
    const std::string& )
{
#if defined(WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE("Reading fields from openPMD files is not supported in 1D");
#elif defined(WARPX_DIM_XZ)
    WARPX_ABORT_WITH_MESSAGE("Reading from openPMD for external fields is not known to work with XZ (see #3828)");
#elif !defined(WARPX_USE_OPENPMD)
    WARPX_ABORT_WITH_MESSAGE("OpenPMD field reading requires OpenPMD support to be enabled");
#endif
}
#endif // WARPX_USE_OPENPMD

void HybridPICModel::CalculateCurrentAmpere (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths)
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateCurrentAmpere(Bfield[lev], edge_lengths[lev], lev);
    }
}

void HybridPICModel::CalculateCurrentAmpere (
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& Bfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& edge_lengths,
    const int lev)
{
    WARPX_PROFILE("WarpX::CalculateCurrentAmpere()");

    auto& warpx = WarpX::GetInstance();
    warpx.get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(
        current_fp_ampere[lev], Bfield, edge_lengths, lev
    );

    // we shouldn't apply the boundary condition to J since J = J_i - J_e but
    // the boundary correction was already applied to J_i and the B-field
    // boundary ensures that J itself complies with the boundary conditions, right?
    // ApplyJfieldBoundary(lev, Jfield[0].get(), Jfield[1].get(), Jfield[2].get());
    for (int i=0; i<3; i++) { current_fp_ampere[lev][i]->FillBoundary(warpx.Geom(lev).periodicity()); }
}

void HybridPICModel::HybridPICSolveE (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> & Efield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Jfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Bfield,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths,
    const bool include_resistivity_term)
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        HybridPICSolveE(
            Efield[lev], Jfield[lev], Bfield[lev], rhofield[lev],
            edge_lengths[lev], lev, include_resistivity_term
        );
    }
}

void HybridPICModel::HybridPICSolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3> & Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& Bfield,
    std::unique_ptr<amrex::MultiFab> const& rhofield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& edge_lengths,
    const int lev, const bool include_resistivity_term)
{
    WARPX_PROFILE("WarpX::HybridPICSolveE()");

    HybridPICSolveE(
        Efield, Jfield, Bfield, rhofield, edge_lengths, lev,
        PatchType::fine, include_resistivity_term
    );
    if (lev > 0)
    {
        amrex::Abort(Utils::TextMsg::Err(
        "HybridPICSolveE: Only one level implemented for hybrid-PIC solver."));
    }
}

void HybridPICModel::HybridPICSolveE (
    std::array< std::unique_ptr<amrex::MultiFab>, 3> & Efield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& Jfield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& Bfield,
    std::unique_ptr<amrex::MultiFab> const& rhofield,
    std::array< std::unique_ptr<amrex::MultiFab>, 3> const& edge_lengths,
    const int lev, PatchType patch_type,
    const bool include_resistivity_term)
{
    auto& warpx = WarpX::GetInstance();

    // Solve E field in regular cells
    warpx.get_pointer_fdtd_solver_fp(lev)->HybridPICSolveE(
        Efield, current_fp_ampere[lev], Jfield, current_fp_external[lev],
        Bfield, bfield_fp_external[lev], 
        rhofield, electron_pressure_fp[lev],
        edge_lengths, lev, this, include_resistivity_term
    );
    warpx.ApplyEfieldBoundary(lev, patch_type);
}

void HybridPICModel::CalculateElectronPressure(DtType a_dt_type)
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        CalculateElectronPressure(lev, a_dt_type);
    }
}

void HybridPICModel::CalculateElectronPressure(const int lev, DtType a_dt_type)
{
    WARPX_PROFILE("WarpX::CalculateElectronPressure()");

    auto& warpx = WarpX::GetInstance();
    // The full step uses rho^{n+1}, otherwise use the old or averaged
    // charge density.
    if (a_dt_type == DtType::Full) {
        FillElectronPressureMF(
            electron_pressure_fp[lev], warpx.getFieldPointer(FieldType::rho_fp, lev)
        );
    } else {
        FillElectronPressureMF(
            electron_pressure_fp[lev], rho_fp_temp[lev].get()
        );
    }
    warpx.ApplyElectronPressureBoundary(lev, PatchType::fine);
    electron_pressure_fp[lev]->FillBoundary(warpx.Geom(lev).periodicity());
}

void HybridPICModel::FillElectronPressureMF (
    std::unique_ptr<amrex::MultiFab> const& Pe_field,
    amrex::MultiFab* const& rho_field ) const
{
    const auto n0_ref = m_n0_ref;
    const auto elec_temp = m_elec_temp;
    const auto gamma = m_gamma;

    // Loop through the grids, and over the tiles within each grid
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        // Extract field data for this grid/tile
        Array4<Real const> const& rho = rho_field->const_array(mfi);
        Array4<Real> const& Pe = Pe_field->array(mfi);

        // Extract tileboxes for which to loop
        const Box& tilebox  = mfi.tilebox();

        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            Pe(i, j, k) = ElectronPressure::get_pressure(
                n0_ref, elec_temp, gamma, rho(i, j, k)
            );
        });
    }
}

void HybridPICModel::BfieldEvolveRK (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Efield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Jfield,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths,
    amrex::Real dt, DtType dt_type,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev)
    {
        BfieldEvolveRK(
            Bfield, Efield, Jfield, rhofield, edge_lengths, dt, lev, dt_type,
            ng, nodal_sync
        );
    }
}

void HybridPICModel::BfieldEvolveRK (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Efield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Jfield,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths,
    amrex::Real dt, int lev, DtType dt_type,
    IntVect ng, std::optional<bool> nodal_sync )
{
    // Make copies of the B-field multifabs at t = n and create multifabs for
    // each direction to store the Runge-Kutta intermediate terms. Each
    // multifab has 2 components for the different terms that need to be stored.
    std::array< MultiFab, 3 > B_old;
    std::array< MultiFab, 3 > K;
    for (int ii = 0; ii < 3; ii++)
    {
        B_old[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 1,
            Bfield[lev][ii]->nGrowVect()
        );
        MultiFab::Copy(B_old[ii], *Bfield[lev][ii], 0, 0, 1, ng);

        K[ii] = MultiFab(
            Bfield[lev][ii]->boxArray(), Bfield[lev][ii]->DistributionMap(), 2,
            Bfield[lev][ii]->nGrowVect()
        );
        K[ii].setVal(0.0);
    }

    // The Runge-Kutta scheme begins here.
    // Step 1:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, edge_lengths,
        0.5_rt*dt, dt_type, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * [-curl x E(B_old)] = B_old + 0.5 * dt * K0.
    for (int ii = 0; ii < 3; ii++)
    {
        // Extract 0.5 * dt * K0 for each direction into index 0 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 0, 1, ng
        );
    }

    // Step 2:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, edge_lengths,
        0.5_rt*dt, dt_type, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K0 + 0.5 * dt * [-curl x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K0 + 0.5 * dt * K1
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K0 from the Bfield for each direction, to get
        // B_new = B_old + 0.5 * dt * K1.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 0, 0, 1, ng);
        // Extract 0.5 * dt * K1 for each direction into index 1 of K.
        MultiFab::LinComb(
            K[ii], 1._rt, *Bfield[lev][ii], 0, -1._rt, B_old[ii], 0, 1, 1, ng
        );
    }

    // Step 3:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, edge_lengths,
        dt, dt_type, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + 0.5 * dt * K1 + dt * [-curl  x E(B_old + 0.5 * dt * K1)]
    //       = B_old + 0.5 * dt * K1 + dt * K2
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract 0.5 * dt * K1 from the Bfield for each direction to get
        // B_new = B_old + dt * K2.
        MultiFab::Subtract(*Bfield[lev][ii], K[ii], 1, 0, 1, ng);
    }

    // Step 4:
    FieldPush(
        Bfield, Efield, Jfield, rhofield, edge_lengths,
        0.5_rt*dt, dt_type, ng, nodal_sync
    );

    // The Bfield is now given by:
    // B_new = B_old + dt * K2 + 0.5 * dt * [-curl x E(B_old + dt * K2)]
    //       = B_old + dt * K2 + 0.5 * dt * K3
    for (int ii = 0; ii < 3; ii++)
    {
        // Subtract B_old from the Bfield for each direction, to get
        // B = dt * K2 + 0.5 * dt * K3.
        MultiFab::Subtract(*Bfield[lev][ii], B_old[ii], 0, 0, 1, ng);

        // Add dt * K2 + 0.5 * dt * K3 to index 0 of K (= 0.5 * dt * K0).
        MultiFab::Add(K[ii], *Bfield[lev][ii], 0, 0, 1, ng);

        // Add 2 * 0.5 * dt * K1 to index 0 of K.
        MultiFab::LinComb(
            K[ii], 1.0, K[ii], 0, 2.0, K[ii], 1, 0, 1, ng
        );

        // Overwrite the Bfield with the Runge-Kutta sum:
        // B_new = B_old + 1/3 * dt * (0.5 * K0 + K1 + K2 + 0.5 * K3).
        MultiFab::LinComb(
            *Bfield[lev][ii], 1.0, B_old[ii], 0, 1.0/3.0, K[ii], 0, 0, 1, ng
        );
    }
}

void HybridPICModel::FieldPush (
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Bfield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>>& Efield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& Jfield,
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const& rhofield,
    amrex::Vector<std::array< std::unique_ptr<amrex::MultiFab>, 3>> const& edge_lengths,
    amrex::Real dt, DtType dt_type,
    IntVect ng, std::optional<bool> nodal_sync )
{
    auto& warpx = WarpX::GetInstance();

    // Calculate J = curl x B / mu0
    CalculateCurrentAmpere(Bfield, edge_lengths);
    // Calculate the E-field from Ohm's law
    HybridPICSolveE(Efield, Jfield, Bfield, rhofield, edge_lengths, true);
    warpx.FillBoundaryE(ng, nodal_sync);
    // Push forward the B-field using Faraday's law
    warpx.EvolveB(dt, dt_type);
    warpx.FillBoundaryB(ng, nodal_sync);
}
