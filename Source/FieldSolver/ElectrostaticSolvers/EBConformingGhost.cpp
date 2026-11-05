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
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#   include <AMReX_MLEBNodeFDLap_K.H>
#   include <AMReX_MultiCutFab.H>
#endif

#include <array>
#include <cmath>
#include <utility>

using namespace amrex::literals;

namespace
{
    // level set value of nodes whose geometry is not available
    constexpr amrex::Real unavailable = amrex::Real(1.e30);

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

    // Geometry with the reach of the lists: a node receives folded charge
    // from covered nodes up to 2 nodes away, whose rule looks 2 nodes further.
    constexpr int ngl = 4;
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

    amrex::Long nfill = 0, nfold = 0, nweights = 0;

    struct Src { amrex::IntVect n; amrex::Real a; };
    struct Rule { amrex::Real b = 0.0_rt; int ns = 0; Src s[2*AMREX_SPACEDIM]; };

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
                    // open length from i toward k
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

        std::vector<amrex::IntVect> fill_node, fill_src, fold_node, fold_src;
        std::vector<amrex::Real> fill_b, fill_w, fold_w;
        std::vector<int> fill_off{0}, fold_off{0};

        amrex::Box const nbx = mfi.validbox();
        Rule R;
        amrex::LoopOnCpu(nbx, [&] (int i, int j, int k)
        {
            amrex::ignore_unused(j, k);
            amrex::IntVect const n(AMREX_D_DECL(i, j, k));
            if (rule(n, R)) {
                fill_node.push_back(n);
                fill_b.push_back(R.b);
                for (int q = 0; q < R.ns; ++q) {
                    fill_src.push_back(R.s[q].n);
                    fill_w.push_back(R.s[q].a);
                }
                fill_off.push_back(static_cast<int>(fill_src.size()));
            }
            if (avail(n) && !covered(n)) {
                std::size_t const before = fold_src.size();
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::IntVect const e = amrex::IntVect::TheDimensionVector(d);
                    for (int s = -1; s <= 1; s += 2) {
                        for (int dist = 1; dist <= 2; ++dist) {
                            amrex::IntVect const kc = n + (s*dist)*e;
                            if (!rule(kc, R)) { continue; }
                            amrex::Real w = 0.0_rt;
                            for (int q = 0; q < R.ns; ++q) {
                                if (R.s[q].n == n) { w += R.s[q].a; }
                            }
                            if (w != 0.0_rt) {
                                fold_src.push_back(kc);
                                fold_w.push_back(w*vol(kc)/vol(n));
                            }
                        }
                    }
                }
                if (fold_src.size() > before) {
                    fold_node.push_back(n);
                    fold_off.push_back(static_cast<int>(fold_src.size()));
                }
            }
        });

        if (fill_node.empty() && fold_node.empty()) { continue; }
        nfill += static_cast<amrex::Long>(fill_node.size());
        nfold += static_cast<amrex::Long>(fold_node.size());
        nweights += static_cast<amrex::Long>(fill_src.size() + fold_src.size());
        BoxLists& B = L.boxes[mfi.index()];
        B.fill_node = ToDevice(fill_node);
        B.fill_b = ToDevice(fill_b);
        B.fill_off = ToDevice(fill_off);
        B.fill_src = ToDevice(fill_src);
        B.fill_w = ToDevice(fill_w);
        B.saved.resize(fill_node.size());
        B.fold_node = ToDevice(fold_node);
        B.fold_off = ToDevice(fold_off);
        B.fold_src = ToDevice(fold_src);
        B.fold_w = ToDevice(fold_w);
        amrex::Gpu::streamSynchronize();
    }
    amrex::ParallelDescriptor::ReduceLongSum({nfill, nfold, nweights});
    auto const sz_iv = amrex::Long(sizeof(amrex::IntVect));
    auto const sz_r = amrex::Long(sizeof(amrex::Real));
    auto const sz_i = amrex::Long(sizeof(int));
    amrex::Long const bytes = nfill*(sz_iv + 3*sz_r + sz_i) + nfold*(sz_iv + sz_i)
        + nweights*(sz_iv + sz_r);
    amrex::Print() << "EB conforming field, level " << lev << ": " << nfill
                   << " covered nodes with ghost values, " << nfold
                   << " open nodes receiving folded charge, " << bytes << " bytes\n";
#else
    amrex::ignore_unused(mf, lev);
#endif
}

void
EBConformingGhost::FoldCharge (amrex::MultiFab& rho, int lev, amrex::Real sign)
{
    ABLASTR_PROFILE("EBConformingGhost::FoldCharge");
    LevelLists& L = GetLists(rho, lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho.nGrowVect().allGE(amrex::IntVect(2)),
        "warpx.eb_e_field = conforming needs 2 ghost cells of rho");
    if (sign > 0.0_rt) {
        rho.FillBoundary(amrex::IntVect(2), WarpX::GetInstance().Geom(lev).periodicity());
    }
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        BoxLists const& B = it->second;
        int const n = static_cast<int>(B.fold_node.size());
        if (n == 0) { continue; }
        auto const* node = B.fold_node.data();
        auto const* off = B.fold_off.data();
        auto const* src = B.fold_src.data();
        auto const* w = B.fold_w.data();
        auto const& r = rho.array(mfi);
        // each entry writes its own open node and reads covered nodes only
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int e) noexcept
        {
            amrex::Real s = 0.0_rt;
            for (int q = off[e]; q < off[e+1]; ++q) { s += w[q]*r(src[q]); }
            r(node[e]) += sign*s;
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
    phi.FillBoundary(amrex::IntVect(2), WarpX::GetInstance().Geom(lev).periodicity());
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        BoxLists& B = it->second;
        int const n = static_cast<int>(B.fill_node.size());
        if (n == 0) { continue; }
        auto const* node = B.fill_node.data();
        auto const* b = B.fill_b.data();
        auto const* off = B.fill_off.data();
        auto const* src = B.fill_src.data();
        auto const* w = B.fill_w.data();
        auto* saved = B.saved.data();
        auto const& p = phi.array(mfi);
        // each entry writes its own covered node and reads open nodes only
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
EBConformingGhost::RestorePhi (amrex::MultiFab& phi, int lev)
{
    ABLASTR_PROFILE("EBConformingGhost::RestorePhi");
    LevelLists& L = GetLists(phi, lev);
#ifdef AMREX_USE_OMP
#   pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto it = L.boxes.find(mfi.index());
        if (it == L.boxes.end()) { continue; }
        BoxLists const& B = it->second;
        int const n = static_cast<int>(B.fill_node.size());
        if (n == 0) { continue; }
        auto const* node = B.fill_node.data();
        auto const* saved = B.saved.data();
        auto const& p = phi.array(mfi);
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int e) noexcept
        {
            p(node[e]) = saved[e];
        });
    }
}
