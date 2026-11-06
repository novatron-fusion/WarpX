/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BOUNDARYHANDLER_H_
#define WARPX_BOUNDARYHANDLER_H_

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLMG.H>
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <array>
#include <string>
#include <utility>


class PoissonBoundaryHandler
{
public:
    PoissonBoundaryHandler (); // constructor

    /** Read runtime parameters. Called in constructor. */
    void ReadParameters ();

    /**
     * \brief Read the input settings and set the boundary conditions used
     * on each domain boundary for the Poisson solver.
     */
    void DefinePhiBCs (const amrex::Geometry& geom);

    /**
     * \brief Initialize amrex::Parser objects to get the boundary potential
     * values at specified times.
     */
    void BuildParsers ();
    void BuildParsersEB ();

    /**
     * \brief Sets the EB potential string and updates the function parser.
     *
     * \param [in] potential The string value of the potential
     */
    void setPotentialEB(const std::string& potential) {
        potential_eb_str = potential;
        BuildParsersEB();
    }

    struct PhiCalculatorEB {

        amrex::Real t;
        amrex::ParserExecutor<4> potential_eb;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real z) const noexcept {
            using namespace amrex::literals;
            return potential_eb(x, 0.0_rt, z, t);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real y, const amrex::Real z) const noexcept {
            return potential_eb(x, y, z, t);
        }
    };

    [[nodiscard]] PhiCalculatorEB
    getPhiEB(amrex::Real t) const noexcept
    {
        return PhiCalculatorEB{t, potential_eb};
    }

    bool m_boundary_potential_specified = false;

    // set default potentials to zero in order for current tests to pass
    // but forcing the user to specify a potential might be better
    std::string potential_xlo_str = "0";
    std::string potential_xhi_str = "0";
    std::string potential_ylo_str = "0";
    std::string potential_yhi_str = "0";
    std::string potential_zlo_str = "0";
    std::string potential_zhi_str = "0";
    std::string potential_eb_str = "0";

    amrex::ParserExecutor<1> potential_xlo;
    amrex::ParserExecutor<1> potential_xhi;
    amrex::ParserExecutor<1> potential_ylo;
    amrex::ParserExecutor<1> potential_yhi;
    amrex::ParserExecutor<1> potential_zlo;
    amrex::ParserExecutor<1> potential_zhi;
    amrex::ParserExecutor<1> potential_eb_t;
    amrex::ParserExecutor<4> potential_eb;

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
    std::array<bool, AMREX_SPACEDIM * 2> dirichlet_flag;
    bool has_non_periodic = false;
    bool phi_EB_only_t = true;

private:

    amrex::Parser potential_xlo_parser;
    amrex::Parser potential_xhi_parser;
    amrex::Parser potential_ylo_parser;
    amrex::Parser potential_yhi_parser;
    amrex::Parser potential_zlo_parser;
    amrex::Parser potential_zhi_parser;
    amrex::Parser potential_eb_parser;
};

/** \brief With an embedded boundary, how E = -grad phi is computed
 *  (warpx.eb_e_field).
 *
 *  Covered nodes lie inside the EB; a cut edge joins an uncovered node to a
 *  covered node, and the EB crosses it at the distance h dx from the
 *  uncovered node.
 *
 *  Gradient: AMReX's EB gradient. On a cut edge it is the one-sided
 *  difference over the length h dx, which amplifies the error of phi at the
 *  uncovered node by 1/h. Where the EB passes close to a node, the field on
 *  that edge, which particles near the wall gather, is then far too large
 *  and heats them.
 *
 *  Centered: the centered difference of WarpX::computeE, with the potential
 *  of the EB on the covered nodes. It is bounded; on the cut edges it is the
 *  field of a wall at the covered node rather than at the EB. With the
 *  symmetric EB Poisson stencil (warpx.eb_poisson_stencil) the force on each
 *  particle is minus the gradient of the electrostatic energy.
 *
 *  Conforming: the centered difference of phi with ghost values on the
 *  covered nodes next to uncovered nodes that extrapolate phi to the EB
 *  potential at the wall, and the charge deposited on those covered nodes
 *  moved to the uncovered nodes with the transposed weights
 *  (EBConformingGhost): the field at the wall to first order, consistent
 *  with the charge assignment. */
enum struct EBEfieldMethod { Gradient, Centered, Conforming };

inline EBEfieldMethod GetEBEfieldMethod ()
{
    static int method = -1;
    if (method < 0) {
        std::string s = "gradient";
        amrex::ParmParse("warpx").query("eb_e_field", s);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            s == "gradient" || s == "centered" || s == "conforming",
            "warpx.eb_e_field must be gradient, centered or conforming");
        method = (s == "gradient") ? 0 : (s == "centered") ? 1 : 2;
    }
    return static_cast<EBEfieldMethod>(method);
}

/** \brief Whether E is computed by WarpX::computeE also with an EB (all but
 *  warpx.eb_e_field = gradient). */
inline bool UseCenteredEBEfield ()
{
    return GetEBEfieldMethod() != EBEfieldMethod::Gradient;
}

/** use amrex to directly calculate the electric field since with EB's the
 *
 * simple finite difference scheme in WarpX::computeE sometimes fails
 */
class EBCalcEfromPhiPerLevel {
    private:
    amrex::Vector<
        amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
    > m_e_field;

    public:
    explicit EBCalcEfromPhiPerLevel(amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> > e_field)
            : m_e_field(std::move(e_field)) {}

    void operator()(amrex::MLMG & mlmg, int const lev) {
        using namespace amrex::literals;

        mlmg.getGradSolution({m_e_field[lev]});
        for (auto &field: m_e_field[lev]) {
            field->mult(-1._rt);
        }
    }
};

#endif // WARPX_BOUNDARYHANDLER_H_
