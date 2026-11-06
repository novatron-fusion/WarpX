/* This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "EBConformingGhost.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_Arena.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#   include <AMReX_MLEBNodeFDLap_K.H>
#   include <AMReX_MultiCutFab.H>
#endif

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace
{
#if defined(AMREX_USE_EB) && (AMREX_SPACEDIM > 1)
    // level set value of nodes whose geometry is not available
    constexpr amrex::Real unavailable = amrex::Real(1.e30);
#endif

    template <typename T>
    amrex::Gpu::DeviceVector<T> ToDevice (std::vector<T> const& h)
    {
        amrex::Gpu::DeviceVector<T> d(h.size());
        if (!h.empty()) {
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(), d.begin());
        }
        return d;
    }
}

EBConformingGhost::LevelLists&
EBConformingGhost::GetLists (amrex::MultiFab const& mf, int lev)
{
    if (static_cast<int>(m_levels.size()) <= lev) { m_levels.resize(lev+1); }
    LevelLists& L = m_levels[lev];
    if (L.ba != mf.boxArray() || L.dm != mf.DistributionMap()) {
        Build(L, mf, lev);
    }
    return L;
}

void
EBConformingGhost::Build (LevelLists& L, amrex::MultiFab const& mf, int lev)
{
    ABLASTR_PROFILE("EBConformingGhost::Build");
    L.ba = mf.boxArray();
    L.dm = mf.DistributionMap();
    L.boxes.clear();
#if defined(AMREX_USE_EB) && (AMREX_SPACEDIM > 1)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mf.ixType().nodeCentered(),
        "warpx.eb_e_field = conforming needs nodal rho and phi");
    auto& warpx = WarpX::GetInstance();
    amrex::Geometry const& geom = warpx.Geom(lev);
    auto const& factory = warpx.fieldEBFactory(lev);
    auto const edgecent = factory.getEdgeCent();
    amrex::MultiFab const& levset_f = factory.getLevelSet();
    amrex::BoxArray const cba = amrex::convert(L.ba, amrex::IntVect(0));

    // Covered nodes within D nodes of an uncovered node, in each direction,
    // get ghost values: the reach of the particle shape. A node receives
    // folded charge from nodes up to 2 nodes away whose rule looks 2 nodes
    // further, and whether a node needs a value depends on the geometry D
    // nodes around it: the reach of the geometry around each box.
    int const D = WarpX::nox/2 + 1;
    constexpr int max_layers = 8;
    int const ngl = 4 + D;
    constexpr int nglay = 4;
    amrex::MFInfo const info = amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    amrex::MultiFab ls(L.ba, L.dm, 1, ngl, info);
    ls.setVal(unavailable);
    amrex::MultiFab::Copy(ls, levset_f, 0, 0, 1,
                          amrex::min(levset_f.nGrowVect(), amrex::IntVect(1)));
    ls.FillBoundary(geom.periodicity());

    // EB position on the edges as in MLEBNodeFDLaplacian: from the low node,
    // 1 if uncut
    std::array<amrex::MultiFab, AMREX_SPACEDIM> pos;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        pos[d].define(amrex::convert(cba, amrex::IntVect::TheEdgeVector(d)),
                      L.dm, 1, ngl-1, info);
        pos[d].setVal(1.0_rt);
        amrex::Dim3 const off = amrex::IntVect::TheDimensionVector(d).dim3();
        for (amrex::MFIter mfi(pos[d]); mfi.isValid(); ++mfi) {
            if (!edgecent[d]->ok(mfi)) { continue; }
            amrex::Box const b = amrex::grow(mfi.validbox(), 1) & (*edgecent[d])[mfi].box();
            auto const& pa = pos[d].array(mfi);
            auto const& eca = edgecent[d]->const_array(mfi);
            auto const& lsa = ls.const_array(mfi);
            amrex::ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::mlebndfdlap_eb_pos_from_cent(i, j, k, pa, eca, lsa, off);
            });
        }
        amrex::Gpu::streamSynchronize();
        pos[d].FillBoundary(geom.periodicity());
    }
    amrex::Gpu::streamSynchronize();

    // nodes inside the domain (periodic directions: all)
    amrex::Box ndomain = amrex::surroundingNodes(geom.Domain());
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (geom.isPeriodic(d)) { ndomain.grow(d, 1000000); }
    }

#if defined(WARPX_DIM_RZ)
    amrex::Real const dr = geom.CellSize(0);
    amrex::Real const rlo = geom.ProbLo(0);
    amrex::Real const axis_f = warpx.UseVerboncoeurAxisCorrection() ? 1.0_rt/3.0_rt : 0.25_rt;
    // node volume of the charge deposition, up to the common factor dr dz
    auto vol = [=] (amrex::IntVect const& n) {
        amrex::Real const r = std::abs(rlo + n[0]*dr);
        return (r == 0.0_rt) ? MathConst::pi*dr*axis_f : 2.0_rt*MathConst::pi*r;
    };
#else
    auto vol = [] (amrex::IntVect const&) { return 1.0_rt; };
#endif

    // Layer map: -1 uncovered, 0 covered without a ghost value, l > 0 covered
    // in layer l (the lists use l - 1). Layer 1: next to an uncovered node
    // along an edge. Layer l > 1: within D nodes of an uncovered node, and in
    // some direction s e the node k+s e is of an earlier layer and k+2s e
    // uncovered or of an earlier layer. Built layer by layer on the valid
    // nodes, with the ghost nodes filled in between.
    amrex::iMultiFab laymf(L.ba, L.dm, 1, nglay, info);
    auto avail_at = [&] (amrex::Array4<amrex::Real const> const& lsa, amrex::IntVect const& n) {
        return ndomain.contains(n) && lsa.contains(n) && lsa(n) < 0.5_rt*unavailable;
    };
    for (amrex::MFIter mfi(laymf); mfi.isValid(); ++mfi) {
        auto const& lsa = ls.const_array(mfi);
        auto const& lay = laymf.array(mfi);
        auto unc = [&] (amrex::IntVect const& m) { return avail_at(lsa, m) && lsa(m) < 0.0_rt; };
        amrex::LoopOnCpu(mfi.fabbox(), [&] (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const m(AMREX_D_DECL(i, j, k));
            int l = 0;
            if (unc(m)) {
                l = -1;
            } else if (avail_at(lsa, m)) {
                for (int d = 0; d < AMREX_SPACEDIM && l == 0; ++d) {
                    amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
                    if (unc(m + e) || unc(m - e)) { l = 1; }
                }
            }
            lay(m) = l;
        });
    }
    // covered within D nodes of an uncovered node
    auto needed_at = [&] (amrex::Array4<amrex::Real const> const& lsa, amrex::IntVect const& m) {
        if (!avail_at(lsa, m) || lsa(m) < 0.0_rt) { return false; }
        bool found = false;
        amrex::LoopOnCpu(amrex::Box(m - amrex::IntVect(D), m + amrex::IntVect(D)),
                         [&] (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const q(AMREX_D_DECL(i, j, k));
            found = found || (avail_at(lsa, q) && lsa(q) < 0.0_rt);
        });
        return found;
    };
    // uncovered, or of an earlier layer than l
    auto before_at = [] (amrex::Array4<int const> const& lay, amrex::IntVect const& m, int l) {
        return lay.contains(m) && (lay(m) == -1 || (lay(m) > 0 && lay(m) < l));
    };
    auto source_dir_at = [&] (amrex::Array4<int const> const& lay, amrex::IntVect const& m,
                              int l, int d, int s) {
        amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
        return before_at(lay, m + s*e, l) && lay(m + s*e) > 0 && before_at(lay, m + (2*s)*e, l);
    };
    for (int l = 2; l <= max_layers; ++l) {
        amrex::Long nadd = 0;
        for (amrex::MFIter mfi(laymf); mfi.isValid(); ++mfi) {
            auto const& lsa = ls.const_array(mfi);
            auto const& lay = laymf.array(mfi);
            amrex::Array4<int const> const layc = laymf.const_array(mfi);
            std::vector<amrex::IntVect> add;
            amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k)
            {
                amrex::ignore_unused(j, k);
                amrex::IntVect const m(AMREX_D_DECL(i, j, k));
                if (layc(m) != 0 || !needed_at(lsa, m)) { return; }
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    for (int s = -1; s <= 1; s += 2) {
                        if (source_dir_at(layc, m, l, d, s)) { add.push_back(m); return; }
                    }
                }
            });
            for (auto const& m : add) { lay(m) = l; }
            nadd += static_cast<amrex::Long>(add.size());
        }
        amrex::ParallelDescriptor::ReduceLongSum(nadd);
        if (nadd == 0) { break; }
        laymf.FillBoundary(geom.periodicity());
    }

    std::vector<amrex::Long> nfill(max_layers, 0), nfold(max_layers, 0);
    amrex::Long nweights = 0, nmissing = 0;
    int nlayers = 0;

    struct Src { amrex::IntVect n; amrex::Real a; };
    struct Rule { amrex::Real b = 0.0_rt; int ns = 0; Src s[4*AMREX_SPACEDIM]; };

    for (amrex::MFIter mfi(mf, amrex::MFItInfo().DisableDeviceSync()); mfi.isValid(); ++mfi)
    {
        auto const& lsa = ls.const_array(mfi);
        std::array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> pa;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { pa[d] = pos[d].const_array(mfi); }

        auto avail = [&] (amrex::IntVect const& n) {
            return ndomain.contains(n) && lsa.contains(n) && lsa(n) < 0.5_rt*unavailable;
        };
        auto covered = [&] (amrex::IntVect const& n) { return lsa(n) >= 0.0_rt; };

        // ghost value of node k: b phi_eb + sum a phi_n
        auto rule = [&] (amrex::IntVect const& k, Rule& R) -> bool
        {
            R = Rule{};
            if (!avail(k) || !covered(k)) { return false; }
            int nedge = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
                for (int s = -1; s <= 1; s += 2) {
                    amrex::IntVect const i = k + s*e;
                    if (!avail(i) || covered(i)) { continue; }
                    // uncovered length from i toward k
                    amrex::Real h = 1.0_rt;
                    amrex::IntVect const lo = (s < 0) ? i : k;
                    if (pa[d].contains(lo)) {
                        amrex::Real const p = pa[d](lo);
                        h = (s < 0) ? p : amrex::mlebndfdlap_hm(p);
                    }
                    h = amrex::min(1.0_rt, amrex::max(0.0_rt, h));
                    ++nedge;
                    if (h >= 0.5_rt) {
                        R.b += 1.0_rt/h;
                        R.s[R.ns++] = Src{i, 1.0_rt - 1.0_rt/h};
                    } else {
                        amrex::IntVect const i2 = i + s*e;
                        if (avail(i2) && !covered(i2)) {
                            R.b += 2.0_rt/(1.0_rt + h);
                            R.s[R.ns++] = Src{i2, -(1.0_rt - h)/(1.0_rt + h)};
                        } else {
                            R.b += 1.0_rt; // the wall at the covered node
                        }
                    }
                }
            }
            if (nedge == 0) { return false; }
            R.b /= nedge;
            for (int q = 0; q < R.ns; ++q) { R.s[q].a /= nedge; }
            return true;
        };

        amrex::Array4<int const> const lay = laymf.const_array(mfi);
        auto in = [&] (amrex::IntVect const& m) { return lay.contains(m); };
        auto needed = [&] (amrex::IntVect const& m) { return needed_at(lsa, m); };
        auto before = [&] (amrex::IntVect const& m, int l) { return before_at(lay, m, l); };
        auto source_dir = [&] (amrex::IntVect const& m, int l, int d, int s) {
            return source_dir_at(lay, m, l, d, s);
        };
        int box_layers = 0;

        // Ghost value of a node k of layer l > 1: in each direction in which
        // k+s e is of an earlier layer and k+2s e uncovered or of an earlier
        // layer, extrapolate linearly from these two; take the mean.
        auto rule_l = [&] (amrex::IntVect const& k, int l, Rule& R) -> bool
        {
            R = Rule{};
            if (!in(k) || lay(k) != l) { return false; }
            if (l == 1) { return rule(k, R); }
            int ndir = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
                for (int s = -1; s <= 1; s += 2) {
                    if (!source_dir(k, l, d, s)) { continue; }
                    R.s[R.ns++] = Src{k + s*e, 2.0_rt};
                    R.s[R.ns++] = Src{k + (2*s)*e, -1.0_rt};
                    ++ndir;
                }
            }
            for (int q = 0; q < R.ns; ++q) { R.s[q].a /= ndir; }
            return ndir > 0;
        };

        struct HostLayer {
            std::vector<amrex::IntVect> fill_node, fill_src, fold_node, fold_src;
            std::vector<amrex::Real> fill_b, fill_w, fold_w;
            std::vector<int> fill_off{0}, fold_off{0};
        };
        std::vector<HostLayer> H(max_layers);

        amrex::Box const nbx = mfi.validbox();
        Rule R;
        amrex::LoopOnCpu(nbx, [&] (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const n(AMREX_D_DECL(i, j, k));
            int const ln = lay(n);
            if (ln > 0 && rule_l(n, ln, R)) {
                HostLayer& h = H[ln-1];
                h.fill_node.push_back(n);
                h.fill_b.push_back(R.b);
                for (int q = 0; q < R.ns; ++q) {
                    h.fill_src.push_back(R.s[q].n);
                    h.fill_w.push_back(R.s[q].a);
                }
                h.fill_off.push_back(static_cast<int>(h.fill_src.size()));
                box_layers = amrex::max(box_layers, ln);
            }
            // The charge of layer l goes to the uncovered nodes and the
            // nodes of the earlier layers it is extrapolated from.
            for (int l = 1; l <= max_layers; ++l) {
                if (!before(n, l)) { continue; }
                HostLayer& h = H[l-1];
                std::size_t const before_size = h.fold_src.size();
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
                    for (int s = -1; s <= 1; s += 2) {
                        for (int dist = 1; dist <= 2; ++dist) {
                            amrex::IntVect const kc = n + (s*dist)*e;
                            if (!rule_l(kc, l, R)) { continue; }
                            amrex::Real w = 0.0_rt;
                            for (int q = 0; q < R.ns; ++q) {
                                if (R.s[q].n == n) { w += R.s[q].a; }
                            }
                            if (w != 0.0_rt) {
                                h.fold_src.push_back(kc);
                                h.fold_w.push_back(w*vol(kc)/vol(n));
                            }
                        }
                    }
                }
                if (h.fold_src.size() > before_size) {
                    h.fold_node.push_back(n);
                    h.fold_off.push_back(static_cast<int>(h.fold_src.size()));
                    box_layers = amrex::max(box_layers, l);
                }
            }
        });

        amrex::LoopOnCpu(nbx, [&] (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const n(AMREX_D_DECL(i, j, k));
            if (lay(n) == 0 && needed(n)) { ++nmissing; }
        });
        if (box_layers == 0) { continue; }
        nlayers = amrex::max(nlayers, box_layers);
        BoxLists& B = L.boxes[mfi.index()];
        B.layer.resize(box_layers);
        for (int l = 0; l < box_layers; ++l) {
            HostLayer const& h = H[l];
            Layer& Y = B.layer[l];
            nfill[l] += static_cast<amrex::Long>(h.fill_node.size());
            nfold[l] += static_cast<amrex::Long>(h.fold_node.size());
            nweights += static_cast<amrex::Long>(h.fill_src.size() + h.fold_src.size());
            Y.fill_node = ToDevice(h.fill_node);
            Y.fill_b = ToDevice(h.fill_b);
            Y.fill_off = ToDevice(h.fill_off);
            Y.fill_src = ToDevice(h.fill_src);
            Y.fill_w = ToDevice(h.fill_w);
            Y.saved.resize(h.fill_node.size());
            Y.fold_node = ToDevice(h.fold_node);
            Y.fold_off = ToDevice(h.fold_off);
            Y.fold_src = ToDevice(h.fold_src);
            Y.fold_w = ToDevice(h.fold_w);
        }
        amrex::Gpu::streamSynchronize();
    }
    amrex::ParallelDescriptor::ReduceIntMax(nlayers);
    amrex::ParallelDescriptor::ReduceLongSum(nfill.data(), max_layers);
    amrex::ParallelDescriptor::ReduceLongSum(nfold.data(), max_layers);
    amrex::ParallelDescriptor::ReduceLongSum(nweights);
    amrex::ParallelDescriptor::ReduceLongSum(nmissing);
    L.nlayers = nlayers;
    amrex::Long nfill_all = 0, nfold_all = 0;
    std::string per_layer;
    for (int l = 0; l < nlayers; ++l) {
        nfill_all += nfill[l];
        nfold_all += nfold[l];
        per_layer += (l == 0 ? "" : " + ") + std::to_string(nfill[l]);
    }
    auto const sz_iv = amrex::Long(sizeof(amrex::IntVect));
    auto const sz_r = amrex::Long(sizeof(amrex::Real));
    auto const sz_i = amrex::Long(sizeof(int));
    amrex::Long const bytes = nfill_all*(sz_iv + 3*sz_r + sz_i) + nfold_all*(sz_iv + sz_i)
        + nweights*(sz_iv + sz_r);
    amrex::Print() << "EB conforming field, level " << lev << ": " << nfill_all
                   << " covered nodes with ghost values in " << nlayers << " layers ("
                   << per_layer << "), " << nfold_all << " lists of folded charge, "
                   << bytes << " bytes\n";
    if (nmissing > 0) {
        amrex::Print() << "EB conforming field, level " << lev << ": " << nmissing
                       << " covered nodes within reach of the particles without a ghost value\n";
    }
#else
    amrex::ignore_unused(mf, lev);
#endif
}

void
EBConformingGhost::ApplyFold (amrex::MultiFab& rho, LevelLists const& L, int layer,
                              amrex::Real sign)
{
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        if (layer >= static_cast<int>(it->second.layer.size())) { continue; }
        Layer const& Y = it->second.layer[layer];
        int const n = static_cast<int>(Y.fold_node.size());
        if (n == 0) { continue; }
        auto const* node = Y.fold_node.data();
        auto const* off = Y.fold_off.data();
        auto const* src = Y.fold_src.data();
        auto const* w = Y.fold_w.data();
        auto const& r = rho.array(mfi);
        // each entry writes its own receiving node and reads covered nodes of
        // this layer only, which no entry of this layer writes
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int e) noexcept
        {
            amrex::Real s = 0.0_rt;
            for (int q = off[e]; q < off[e+1]; ++q) { s += w[q]*r(src[q]); }
            r(node[e]) += sign*s;
        });
    }
}

void
EBConformingGhost::FoldCharge (amrex::MultiFab& rho, int lev, amrex::Real sign)
{
    ABLASTR_PROFILE("EBConformingGhost::FoldCharge");
    LevelLists& L = GetLists(rho, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.nGrowVect().allGE(amrex::IntVect(2)),
        "warpx.eb_e_field = conforming needs 2 ghost cells of rho");
    auto const period = WarpX::GetInstance().Geom(lev).periodicity();
    // Transpose of FillGhost: each layer is filled from the earlier ones, so
    // the charge of the last layer is folded first, onto the earlier layers
    // and the uncovered nodes, and so on down to the first layer. sign = -1
    // undoes this in the reverse order.
    if (sign > 0.0_rt) {
        rho.FillBoundary(amrex::IntVect(2), period);
        for (int l = L.nlayers - 1; l >= 0; --l) {
            ApplyFold(rho, L, l, sign);
            if (l > 0) { rho.FillBoundary(amrex::IntVect(2), period); }
        }
    } else {
        for (int l = 0; l < L.nlayers; ++l) { ApplyFold(rho, L, l, sign); }
    }
}

void
EBConformingGhost::ApplyFill (amrex::MultiFab& phi, LevelLists& L, int layer)
{
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        if (layer >= static_cast<int>(it->second.layer.size())) { continue; }
        Layer& Y = it->second.layer[layer];
        int const n = static_cast<int>(Y.fill_node.size());
        if (n == 0) { continue; }
        auto const* node = Y.fill_node.data();
        auto const* b = Y.fill_b.data();
        auto const* off = Y.fill_off.data();
        auto const* src = Y.fill_src.data();
        auto const* w = Y.fill_w.data();
        auto* saved = Y.saved.data();
        auto const& p = phi.array(mfi);
        // each entry writes its own covered node and reads nodes that no
        // entry of this layer writes
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int e) noexcept
        {
            amrex::Real const peb = p(node[e]);
            saved[e] = peb;
            amrex::Real v = b[e]*peb;
            for (int q = off[e]; q < off[e+1]; ++q) { v += w[q]*p(src[q]); }
            p(node[e]) = v;
        });
    }
}

void
EBConformingGhost::FillGhost (amrex::MultiFab& phi, int lev)
{
    ABLASTR_PROFILE("EBConformingGhost::FillGhost");
    LevelLists& L = GetLists(phi, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(phi.nGrowVect().allGE(amrex::IntVect(2)),
        "warpx.eb_e_field = conforming needs 2 ghost cells of phi");
    auto const period = WarpX::GetInstance().Geom(lev).periodicity();
    phi.FillBoundary(amrex::IntVect(2), period);
    for (int l = 0; l < L.nlayers; ++l) {
        // a layer reads the values of the earlier layers in the neighboring boxes
        if (l > 0) { phi.FillBoundary(amrex::IntVect(2), period); }
        ApplyFill(phi, L, l);
    }
}

void
EBConformingGhost::ApplyRestore (amrex::MultiFab& phi, LevelLists const& L, int layer)
{
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        if (layer >= static_cast<int>(it->second.layer.size())) { continue; }
        Layer const& Y = it->second.layer[layer];
        int const n = static_cast<int>(Y.fill_node.size());
        if (n == 0) { continue; }
        auto const* node = Y.fill_node.data();
        auto const* saved = Y.saved.data();
        auto const& p = phi.array(mfi);
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int e) noexcept
        {
            p(node[e]) = saved[e];
        });
    }
}

void
EBConformingGhost::RestorePhi (amrex::MultiFab& phi, int lev)
{
    ABLASTR_PROFILE("EBConformingGhost::RestorePhi");
    LevelLists& L = GetLists(phi, lev);
    for (int l = L.nlayers - 1; l >= 0; --l) { ApplyRestore(phi, L, l); }
    if (L.nlayers > 1) {
        // the ghost cells hold ghost values of the earlier layers from FillGhost
        phi.FillBoundary(amrex::IntVect(2), WarpX::GetInstance().Geom(lev).periodicity());
    }
}
