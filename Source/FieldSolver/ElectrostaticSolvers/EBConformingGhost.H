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
 * A particle next to the wall also interpolates from covered nodes that have
 * no uncovered neighbor along an edge: the other corners of its cell, and
 * with particle shapes above 1 nodes farther in. The covered nodes within
 * D = shape/2 + 1 nodes (in each direction) of an uncovered node get ghost
 * values too, in later layers: such a node extrapolates linearly along an
 * edge from a node k1 of an earlier layer and the node k2 beyond it,
 * uncovered or of an earlier layer, phi_k = 2 phi_k1 - phi_k2, averaged over
 * the directions where this is possible. Without them the field of such a
 * particle is interpolated partly from edges inside the EB and can be
 * several times too small. The layers are filled in order, and their charge
 * is folded in the reverse order.
 *
 * The weights depend on the geometry only. They are built once per level
 * into per-box lists for each layer: one entry per covered node with a ghost
 * value and one per node that receives folded charge (uncovered, or of an
 * earlier layer), so the memory is proportional to the number of nodes near
 * the EB.
 */
class EBConformingGhost
{
public:
    /** Add (sign = 1) or remove (sign = -1) the charge of the covered nodes
     *  folded onto the uncovered nodes of rho (nodal, at least 2 ghost
     *  cells). */
    void FoldCharge (amrex::MultiFab& rho, int lev, amrex::Real sign);

    /** Replace phi on the covered nodes of all layers by the ghost values
     *  (phi needs 2 ghost cells); RestorePhi puts the EB potential back. */
    void FillGhost (amrex::MultiFab& phi, int lev);
    void RestorePhi (amrex::MultiFab& phi, int lev);

    // public because Build and the Apply functions launch device lambdas (CUDA)
    struct Layer {
        // ghost values: node, b, and CSR list of (source node, weight)
        amrex::Gpu::DeviceVector<amrex::IntVect> fill_node;
        amrex::Gpu::DeviceVector<amrex::Real> fill_b;
        amrex::Gpu::DeviceVector<int> fill_off;
        amrex::Gpu::DeviceVector<amrex::IntVect> fill_src;
        amrex::Gpu::DeviceVector<amrex::Real> fill_w;
        amrex::Gpu::DeviceVector<amrex::Real> saved;
        // folding: receiving node and CSR list of (covered node, weight)
        amrex::Gpu::DeviceVector<amrex::IntVect> fold_node;
        amrex::Gpu::DeviceVector<int> fold_off;
        amrex::Gpu::DeviceVector<amrex::IntVect> fold_src;
        amrex::Gpu::DeviceVector<amrex::Real> fold_w;
    };
    // layer 0: covered nodes next to an uncovered node along an edge; layer
    // l > 0: covered nodes extrapolated from the layers before
    struct BoxLists {
        std::vector<Layer> layer;
    };
    struct LevelLists {
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        std::map<int, BoxLists> boxes;
        int nlayers = 0;
    };

    LevelLists& GetLists (amrex::MultiFab const& mf, int lev);
    void Build (LevelLists& L, amrex::MultiFab const& mf, int lev);
    static void ApplyFill (amrex::MultiFab& phi, LevelLists& L, int layer);
    static void ApplyRestore (amrex::MultiFab& phi, LevelLists const& L, int layer);
    static void ApplyFold (amrex::MultiFab& rho, LevelLists const& L, int layer, amrex::Real sign);

private:
    std::vector<LevelLists> m_levels;
};

#endif // WARPX_EB_CONFORMING_GHOST_H_
