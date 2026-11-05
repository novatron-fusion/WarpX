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
 * The covered nodes next to the fluid get ghost values that extrapolate phi
 * linearly from the open nodes to the EB potential at the wall,
 *   phi_k = b_k phi_eb + sum_m a_km phi_m ,
 * and the centered difference of these values is the field the particles
 * gather. The charge that the particles deposit on those covered nodes is
 * folded onto the open nodes with the transposed weights a_km V_k / V_m
 * before the Poisson solve. Charge assignment and field interpolation then
 * remain transposes of each other, so with a symmetric Poisson operator
 * (warpx.eb_poisson_stencil = symmetric) the potential is reciprocal and the
 * forces derive from the electrostatic energy, while the field on a cut edge
 * is the slope to the wall rather than to the covered node.
 *
 * Along the edge from an open node i to a covered node k, with the EB at the
 * distance h dx from i: for h >= 1/2, phi_k = phi_i + (phi_eb - phi_i)/h; for
 * h < 1/2 the line through the next open node i2 (opposite k) and the wall,
 * phi_k = phi_eb + (phi_eb - phi_i2)(1 - h)/(1 + h), so that the error of
 * phi enters with a factor of at most 2. A covered node next to several open
 * nodes gets the mean of the edge extrapolations. In 1D, and for a wall that
 * cuts each node in one direction only, this is the conforming linear finite
 * element method with the last element shortened to h dx, whose stiffness
 * matrix is the symmetric EB stencil.
 *
 * The weights depend on the geometry only. They are built once per level
 * into per-box lists: one entry per covered node with ghost values and one
 * per open node that receives folded charge, so the memory is proportional
 * to the number of nodes next to the EB.
 */
class EBConformingGhost
{
public:
    /** Add (sign = 1) or remove (sign = -1) the folded charge of the covered
     *  nodes to the open nodes of rho (nodal, at least 2 ghost cells). */
    void FoldCharge (amrex::MultiFab& rho, int lev, amrex::Real sign);

    /** Replace phi on the covered nodes next to the fluid by the ghost values
     *  (phi needs 2 ghost cells); RestorePhi puts the EB potential back. */
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
        // folding: open node and CSR list of (covered node, weight)
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
