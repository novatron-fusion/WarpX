#ifndef WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H
#define WARPX_SOURCE_EMBEDDEDBOUNDARY_WARPXPOLYGONIF_H

#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_EB2_IF_Base.H>

#include <cmath>
#include <algorithm>
#include <regex>

// For all implicit functions, >0: body; =0: boundary; <0: fluid

struct Point2D {
    amrex::Real r, z;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Point2D(amrex::Real r_, amrex::Real z_) : r(r_), z(z_) {}
};

typedef std::vector<Point2D> Curve;

Curve parse_curve(std::vector<std::string> strarr)
{
  std::string float_regex = "[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?";
  std::regex exp(" *\\( *(" + float_regex + ") *, *(" + float_regex + ") *\\) *");

  Curve curve;

  for(auto str: strarr) {
    std::smatch res;
    if(!std::regex_match(str, res, exp))
      throw std::invalid_argument("cannot parse pair of floats");
    double r, z;
    // res[0] is the entire match
    r = std::stod(res[1]);
    // res[2] is the optional exponent for the first number
    z = std::stod(res[3]);
    // res[4] is the optional exponent for the second number
    curve.push_back(Point2D((amrex::Real)r, (amrex::Real)z));
  }
  return curve;
}

class PolygonXYIF
    : amrex::GPUable
{
public:

    PolygonXYIF (Curve curve, bool a_inside)
        : m_curve(curve),
          m_inside(a_inside)
        {}

    ~PolygonXYIF () {}
    
    PolygonXYIF (const PolygonXYIF& rhs) noexcept = default;
    PolygonXYIF (PolygonXYIF&& rhs) noexcept = default;
    PolygonXYIF& operator= (const PolygonXYIF& rhs) = delete;
    PolygonXYIF& operator= (PolygonXYIF&& rhs) = delete;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
        const noexcept
    {
        amrex::Real dist = 1;
        for (size_t i = 0; i < m_curve.size() - 1; i++)
            dist *= intersects_outwards(Point2D(x, y), i);
        return dist;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator() (const amrex::RealArray& p) const noexcept
    {
        return this->operator() (AMREX_D_DECL(p[0], p[1], p[2]));
    }

protected:
    Curve            m_curve;
    bool             m_inside;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real intersects_outwards(Point2D p, int lineseg_i) const noexcept
    {
        // Determine if the point is radially inside the line segment from curve[lineseg_i] to curve[lineseg_i+1].

        Point2D a = m_curve[lineseg_i];
        Point2D b = m_curve[lineseg_i + 1];

        if (a.r <= p.r && b.r <= p.r) // point is completely outside the linesegment
            return 1; // not inside
        if (a.z < p.z && b.z < p.z) // point is completely above the linesegment
            return 1; // not inside
        if (a.z >= p.z && b.z >= p.z) // point is completely below the linesegment
            return 1; // not inside
        if (a.r > p.r && b.r > p.r && a.z < p.z && p.z <= b.z) // point is completely inside the linesegment
            return -1; // inside

        // no clear bounding box decision can be made. Check for real if the point lies to the left or right of the line segment

        auto p_minus_a = Point2D(p.r - a.r, p.z - a.z);
        auto b_minus_a = Point2D(b.r - a.r, b.z - a.z);

        // check the sign of the third component of the cross product of (p-a) x (b-a)

        auto cross = (p_minus_a.r * b_minus_a.z) - (p_minus_a.z * b_minus_a.r);
        if (a.z < b.z)
            return cross;
        else
            return -cross;
    }
};

#endif