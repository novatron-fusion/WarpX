#ifndef WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H
#define WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H
#include "WarpX.H"
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_EB2_IF_Base.H>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <regex>
#include <boost/geometry/io/wkt/read.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

typedef boost::geometry::model::d2::point_xy<amrex::Real> point_t;
typedef boost::geometry::model::polygon<point_t> polygon_t;
typedef boost::geometry::model::multi_polygon<polygon_t> mpolygon_t;

namespace
{
    // For all implicit functions, >0: body; =0: boundary; <0: fluid
    inline void add_jump(amrex::Gpu::DeviceVector<amrex::Real> &r, amrex::Gpu::DeviceVector<amrex::Real> &z, amrex::Gpu::DeviceVector<size_t> &jumps)
    {
        if (jumps.size() > 0 && jumps[jumps.size() - 1] == static_cast<size_t>(r.size() - 1))
        {
            r.pop_back();
            z.pop_back();
        }
        else
        {
            jumps.push_back(r.size()); 
        }
    }

    template <typename T>
    inline void expand_ring(const T &polygon, amrex::Gpu::DeviceVector<amrex::Real> &r, amrex::Gpu::DeviceVector<amrex::Real> &z, amrex::Gpu::DeviceVector<size_t> &jumps)
    {
        for (auto p : polygon)
        {
            // If we are at the z axis we want to remove all those points, otherwise they
            // will duplicate when we rotate the polygon around the z axis.
            if ((p.x() == 0) && (r.size() > 0) && (r[r.size() - 1] == 0)){ 
                std::cout << "p.x(): " << p.x() << ", r.size(): " << r.size() << ", r[r.size() - 1]: " << r[r.size() - 1] << std::endl;
                std::cout << "Adding jump inside loop" << std::endl;
                add_jump(r, z, jumps);
                std::cout << "After adding jump inside loop" << std::endl;
                for (size_t idx = 0; idx < jumps.size(); idx++) {
                    std::cout << "jumps[" << idx << "]: " << jumps[idx] << std::endl;
                }
            }
            //std::cout << "Adding point: (" << p.x() << ", " << p.y() << ")" << std::endl;
            r.push_back(p.x());
            z.push_back(p.y());
        }

    }

    void expand_multipolygon(const mpolygon_t &multipolygon, amrex::Gpu::DeviceVector<amrex::Real> &r, amrex::Gpu::DeviceVector<amrex::Real> &z, amrex::Gpu::DeviceVector<size_t> &jumps)
    {
        bool first = true;
        for (auto polygon : multipolygon)
        {
            std::cout << "Expanding polygon with " << polygon.outer().size() << " outer points and " << polygon.inners().size() << " inner rings." << std::endl;
            if (!first) {
                add_jump(r, z, jumps); // Add a jump before the next polygon
            }
            expand_ring(polygon.outer(), r, z, jumps);

            for (auto ring : polygon.inners()){
                add_jump(r, z, jumps); 
                expand_ring(ring, r, z, jumps);
            }
            first = false;       
        }
    }
    void parse_multipolygon(const std::string &wkt, amrex::Gpu::DeviceVector<amrex::Real> &r, amrex::Gpu::DeviceVector<amrex::Real> &z, amrex::Gpu::DeviceVector<size_t> &jumps) {
        mpolygon_t multipolygon;
        boost::geometry::read_wkt(wkt, multipolygon);

        expand_multipolygon(multipolygon, r, z, jumps);
        std::cout << "Parsed multipolygon with " << r.size() << " points." << std::endl;
        std::cout << "Jumps: ";
        for (size_t idx = 0; idx < jumps.size(); idx++) {
            std::cout << jumps[idx] << " ";
        }
        std::cout << std::endl;
    }
    class PolygonXYIF
        : public amrex::GPUable
    {
    public:
        PolygonXYIF(amrex::Gpu::DeviceVector<amrex::Real> &r, amrex::Gpu::DeviceVector<amrex::Real> &z, amrex::Gpu::DeviceVector<size_t> &jumps)
            : m_r(r.data()),
              m_z(z.data()),
              m_jumps(jumps.data()),
              m_size(r.size()),
              m_jumps_size(jumps.size())
        {
            assert(r.size() == z.size());
        }
        ~PolygonXYIF() {}
        PolygonXYIF(const PolygonXYIF &rhs) noexcept = default;
        PolygonXYIF(PolygonXYIF &&rhs) noexcept = default;
        PolygonXYIF &operator=(const PolygonXYIF &rhs) = delete;
        PolygonXYIF &operator=(PolygonXYIF &&rhs) = delete;
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
            const noexcept
        {
            amrex::Real dist = 1;
            size_t i_jump = 0;
            for (size_t i = 0; i < m_size - 1; i++) {
                if(i_jump < m_jumps_size && m_jumps[i_jump] == i+1) {
                    i_jump++;
                    continue;
                }
                dist *= intersects_outwards(x, y, i);
            }
            if (std::abs(y) < 1.5 && std::abs(x) < 1) {
                std::cout << "PolygonXYIF::operator() called with point (" << x << ", " << y << "), dist: " << dist << '\n';
            }
            return dist;
        }
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        operator()(const amrex::RealArray &p) const noexcept
        {
            return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
        }
    protected:
        amrex::Real const* m_r;
        amrex::Real const* m_z;
        size_t const* m_jumps;
        size_t m_size;
        size_t m_jumps_size;
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
            amrex::Real
            intersects_outwards(amrex::Real r, amrex::Real z, size_t i) const noexcept
        {
            // Determine if the point is radially inside the line segment from curve[lineseg_i] to curve[lineseg_i+1].
            if (m_r[i] <= r && m_r[i + 1] <= r)                                // point is completely outside the linesegment
                return 1;                                                      // not inside
            if (m_z[i] < z && m_z[i + 1] < z)                                  // point is completely above the linesegment
                return 1;                                                      // not inside
            if (m_z[i] >= z && m_z[i + 1] >= z)                                // point is completely below the linesegment
                return 1;                                                      // not inside
            if (m_r[i] > r && m_r[i + 1] > r && m_z[i] < z && z <= m_z[i + 1]) // point is completely inside the linesegment
                return -1;                                                     // inside
            // no clear bounding box decision can be made. Check for real if the point lies to the left or right of the line segment
            auto p_minus_a_r = r - m_r[i];
            auto p_minus_a_z = z - m_z[i];
            auto b_minus_a_r = m_r[i + 1] - m_r[i];
            auto b_minus_a_z = m_z[i + 1] - m_z[i];
            // check the sign of the third component of the cross product of (p-a) x (b-a)
            auto cross = (p_minus_a_r * b_minus_a_z) - (p_minus_a_z * b_minus_a_r);
            if (m_z[i] < m_z[i + 1])
                return cross;
            else
                return -cross;
        }
    };
}
#endif
