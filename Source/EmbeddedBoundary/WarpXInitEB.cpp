/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#  include "Fields.H"
#  include "Utils/Parser/ParserUtils.H"
#  include "Utils/TextMsg.H"
#  include "PolygonXYIF.H"

#   include <AMReX_BLProfiler.H>
#   include <AMReX_BoxArray.H>
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB2_IF.H>
#   include <AMReX_EB2_IF_Base.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_GpuQualifiers.H>
#   include <AMReX_ParmParse.H>
#   include <AMReX_REAL.H>
#   include <AMReX_SPACE.H>

#  include <cstdlib>
#  include <string>

using namespace ablastr::fields;

#endif

#ifdef AMREX_USE_EB
namespace {
    class ParserIF
        : public amrex::GPUable
    {
    public:
        ParserIF (const amrex::ParserExecutor<3>& a_parser)
            : m_parser(a_parser)
            {}

        ParserIF (const ParserIF& rhs) noexcept = default;
        ParserIF (ParserIF&& rhs) noexcept = default;
        ParserIF& operator= (const ParserIF& rhs) = delete;
        ParserIF& operator= (ParserIF&& rhs) = delete;

        ~ParserIF() = default;

        AMREX_GPU_HOST_DEVICE inline
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return m_parser(x,amrex::Real(0.0),y);
#else
            return m_parser(x,y,z);
#endif
        }

        inline amrex::Real operator() (const amrex::RealArray& p) const noexcept {
            return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
        }

    private:
        amrex::ParserExecutor<3> m_parser; //! function parser with three arguments (x,y,z)
    };
}
#endif

void
WarpX::InitEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("InitEB only works when EBs are enabled at runtime");
    }

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("EBs only implemented in 2D and 3D");
#endif

#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    const amrex::ParmParse pp_warpx("warpx");
    std::string impf;
    pp_warpx.query("eb_implicit_function", impf);
    std::string wkt_file;
    pp_warpx.query("wkt_file", wkt_file);
    if (! impf.empty()) {
        auto eb_if_parser = utils::parser::makeParser(impf, {"x", "y", "z"});
        ParserIF const pif(eb_if_parser.compile<3>());
        auto gshop = amrex::EB2::makeShop(pif, eb_if_parser);
         // The last argument of amrex::EB2::Build is the maximum coarsening level
         // to which amrex should try to coarsen the EB.  It will stop after coarsening
         // as much as it can, if it cannot coarsen to that level.  Here we use a big
         // number (e.g., maxLevel()+20) for multigrid solvers.  Because the coarse
         // level has only 1/8 of the cells on the fine level, the memory usage should
         // not be an issue.
        amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
    } else if(!wkt_file.empty()) {
        std::ifstream wkt_multipolygon_file(wkt_file);
        std::string wkt_multipolygon(std::istreambuf_iterator<char>{wkt_multipolygon_file}, {});
        amrex::Vector<amrex::Real> r_vec(0), z_vec(0);
        amrex::Vector<size_t> jump_vec(0);
        parse_multipolygon(wkt_multipolygon, r_vec, z_vec, jump_vec);
        amrex::Real *r_data, *z_data;
        size_t *jump_data;
#ifdef AMREX_USE_GPU
        amrex::Gpu::DeviceVector<amrex::Real> r_gpuvec(r_vec.size());
        amrex::Gpu::DeviceVector<amrex::Real> z_gpuvec(z_vec.size());
        amrex::Gpu::DeviceVector<size_t> jump_gpuvec(jump_vec.size());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, r_vec.begin(), r_vec.end(), r_gpuvec.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, z_vec.begin(), z_vec.end(), z_gpuvec.begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, jump_vec.begin(), jump_vec.end(), jump_gpuvec.begin());
        amrex::Gpu::synchronize();
        r_data = r_gpuvec.data();
        z_data = z_gpuvec.data();
        jump_data = jump_gpuvec.data();
#else
        r_data = r_vec.data();
        z_data = z_vec.data();
        jump_data = jump_data.data();
#endif
        PolygonXYIF polygonXY(r_data, z_data, r_vec.size(), jump_data, jump_vec.size());
        auto latheif = amrex::EB2::lathe(polygonXY);
        auto gshop = amrex::EB2::makeShop(latheif, polygonXY);
        amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
    } else {
        amrex::ParmParse pp_eb2("eb2");
        if (!pp_eb2.contains("geom_type")) {
            std::string const geom_type = "all_regular";
            pp_eb2.add("geom_type", geom_type); // use all_regular by default
        }
        // See the comment above on amrex::EB2::Build for the hard-wired number 20.
        amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel()+20);
    }
#endif
}

void
WarpX::ComputeDistanceToEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeDistanceToEB only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeDistanceToEB");
    using warpx::fields::FieldType;
    const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
    for (int lev=0; lev<=maxLevel(); lev++) {
        const amrex::EB2::Level& eb_level = eb_is.getLevel(Geom(lev));
        auto const eb_fact = fieldEBFactory(lev);
        amrex::FillSignedDistance(*m_fields.get(FieldType::distance_to_eb, lev), eb_level, eb_fact, 1);
    }
#endif
}
