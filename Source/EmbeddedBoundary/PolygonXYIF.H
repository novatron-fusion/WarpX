#ifndef WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H
#define WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H

#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_EB2_IF_Base.H>

#include <cmath>
#include <algorithm>
#include <regex>

namespace
{
    // For all implicit functions, >0: body; =0: boundary; <0: fluid

    void parse_curve(std::vector<std::string> strarr, amrex::Vector<amrex::Real> &r_vec, amrex::Vector<amrex::Real> &z_vec)
    {
        std::string float_regex = "[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?";
        std::regex exp(" *\\( *(" + float_regex + ") *, *(" + float_regex + ") *\\) *");

        for (auto str : strarr)
        {
            std::smatch res;
            if (!std::regex_match(str, res, exp))
                throw std::invalid_argument("cannot parse pair of floats");
            double r, z;
            // res[0] is the entire match
            r = std::stod(res[1]);
            // res[2] is the optional exponent for the first number
            z = std::stod(res[3]);
            // res[4] is the optional exponent for the second number
            r_vec.push_back((amrex::Real)r);
            z_vec.push_back((amrex::Real)z);
        }
    }

    class PolygonXYIF
        : public amrex::GPUable
    {
    public:
        PolygonXYIF(amrex::Real* r, amrex::Real *z, size_t n)
            : m_r(r),
              m_z(z),
              m_n(n)
        {
        }

        ~PolygonXYIF() {}

        PolygonXYIF(const PolygonXYIF &rhs) noexcept = default;
        PolygonXYIF(PolygonXYIF &&rhs) noexcept = default;
        PolygonXYIF &operator=(const PolygonXYIF &rhs) = delete;
        PolygonXYIF &operator=(PolygonXYIF &&rhs) = delete;

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        operator()(AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
            const noexcept
        {
            amrex::Real dist = 1;

            for (size_t i = 0; i < m_n - 1; i++)
                dist *= intersects_outwards(x, y, i);

            return dist;
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        amrex::Real
        operator()(const amrex::RealArray &p) const noexcept
        {
            return this->operator()(AMREX_D_DECL(p[0], p[1], p[2]));
        }

    protected:
        amrex::Real *m_r;
        amrex::Real *m_z;
        size_t m_n;

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
            amrex::Real
            intersects_outwards(amrex::Real r, amrex::Real z, int i) const noexcept
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