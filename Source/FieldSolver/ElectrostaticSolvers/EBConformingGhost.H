/* This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EB_CONFORMING_GHOST_H_
#define WARPX_EB_CONFORMING_GHOST_H_

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <map>
#include <vector>

/**
 * \brief Wall-conforming charge assignment and field interpolation at an
 * embedded boundary (warpx.eb_e_field = conforming).
 *
 * A node is covered if it lies inside the EB, and uncovered otherwise; a cut
 * edge joins an uncovered node to a covered node. The covered nodes next to
 * uncovered nodes get ghost values that extrapolate phi linearly from the
 * uncovered nodes m to the EB potential phi_eb at the wall,
 *   phi_k = b_k phi_eb + sum_m a_km phi_m ,
 * and the centered difference of these values is the field the particles
 * gather. Before the Poisson solve, the charge that the particles deposit on
 * those covered nodes is moved to the uncovered nodes with the transposed
 * weights a_km V_k / V_m, where V is the node volume ("folding"). Charge
 * assignment and field interpolation then remain transposes of each other,
 * so with a symmetric Poisson operator (warpx.eb_poisson_stencil =
 * symmetric) the potential is reciprocal and the force on each particle is
 * minus the gradient of the electrostatic energy. The field on a cut edge is
 * (phi_eb - phi_i) / (h dx), the difference to the wall rather than to the
 * covered node.
 *
 * Along the cut edge from an uncovered node i to a covered node k, with the
 * EB at the distance h dx from i: for h >= 1/2, phi_k = phi_i + (phi_eb -
 * phi_i)/h; for h < 1/2 the line through the next uncovered node i2
 * (opposite k) and the wall, phi_k = phi_eb + (phi_eb - phi_i2)(1 - h)/(1 + h),
 * so that the error of phi enters with a factor of at most 2. A covered node
 * next to several uncovered nodes gets the mean of the edge extrapolations.
 * In 1D, and for a wall that cuts each node in one direction only, this is
 * the conforming linear finite element method with the last element
 * shortened to h dx, whose stiffness matrix is the symmetric EB stencil.
 *
 * The weights depend on the geometry only. They are built once per level
 * into per-box lists: one entry per covered node with ghost values and one
 * per uncovered node that receives folded charge, so the memory is
 * proportional to the number of nodes next to the EB.
 */
class EBConformingGhost
{
public:
    /** Add (sign = 1) or remove (sign = -1) the charge of the covered nodes
     *  folded onto the uncovered nodes of rho (nodal, at least 2 ghost
     *  cells). */
    void FoldCharge (amrex::MultiFab& rho, int lev, amrex::Real sign);

    /** Replace phi on the covered nodes next to uncovered nodes by the ghost
     *  values (phi needs 2 ghost cells); RestorePhi puts the EB potential
     *  back. */
    void FillGhost (amrex::MultiFab& phi, int lev);
    void RestorePhi (amrex::MultiFab& phi, int lev);

    // public because Build launches device lambdas (CUDA)
    struct BoxLists {
        // ghost values: node, b, and CSR list of (source node, weight)
        amrex::Gpu::DeviceVector<amrex::IntVect> fill_node;
        amrex::Gpu::DeviceVector<amrex::Real> fill_b;
        amrex::Gpu::DeviceVector<int> fill_off;
        amrex::Gpu::DeviceVector<amrex::IntVect> fill_src;
        amrex::Gpu::DeviceVector<amrex::Real> fill_w;
        amrex::Gpu::DeviceVector<amrex::Real> saved;
        // folding: uncovered node and CSR list of (covered node, weight)
        amrex::Gpu::DeviceVector<amrex::IntVect> fold_node;
        amrex::Gpu::DeviceVector<int> fold_off;
        amrex::Gpu::DeviceVector<amrex::IntVect> fold_src;
        amrex::Gpu::DeviceVector<amrex::Real> fold_w;
    };
    struct LevelLists {
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        std::map<int, BoxLists> boxes;
    };

    LevelLists& GetLists (amrex::MultiFab const& mf, int lev);
    void Build (LevelLists& L, amrex::MultiFab const& mf, int lev);

private:
    std::vector<LevelLists> m_levels;
};

#endif // WARPX_EB_CONFORMING_GHOST_H_
