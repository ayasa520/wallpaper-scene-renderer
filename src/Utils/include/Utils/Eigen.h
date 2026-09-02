#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace Eigen
{

constexpr double Radians(double a) noexcept { return (a / 180.0f) * (double)EIGEN_PI; }

Matrix4d inline LookAt(Vector3d eye, Vector3d center, Vector3d up) noexcept {
    Vector3d camDir = center - eye;
    // cam look at neg z
    Vector3d zAxis = -camDir.normalized();
    Vector3d xAxis = up.cross(zAxis).normalized();
    Vector3d yAxis = zAxis.cross(xAxis).normalized();

    Affine3d trans = Affine3d::Identity();
    // World-to-camera basis vectors must occupy rows for Eigen's column-vector convention:
    // camera.x = dot(world - eye, xAxis), camera.y = dot(world - eye, yAxis),
    // camera.z = dot(world - eye, zAxis). Columns accidentally build the transpose, which is mostly
    // invisible for axis-aligned 2D cameras but pushes authored 3D camera paths off target.
    trans.linear().row(0) = xAxis.transpose();
    trans.linear().row(1) = yAxis.transpose();
    trans.linear().row(2) = zAxis.transpose();

    // translate
    trans *= Translation3d(-eye);

    return trans.matrix();
}

// Scene depth is reversed: both projections place the near plane at clip depth 1 and the far
// plane at 0. Depth buffers are cleared to 0 and tested with GREATER. With a floating-point
// depth attachment this keeps near-uniform relative precision along the whole view distance, so
// concentric shells a fraction of a percent apart (planet surface, clouds, atmosphere) still
// resolve when the camera is tens of thousands of near-plane distances away.
Matrix4d inline Ortho(double left, double right, double bottom, double top, double nearz,
                      double farz) noexcept {
    // Right-handed view space looks down -z: a point at distance d in front of the eye has
    // view z = -d and maps to (farz - d) / (farz - nearz).
    Matrix4d m   = Matrix4d::Identity();
    m(0, 0)      = 2.0 / (right - left);
    m(0, 3)      = -(right + left) / (right - left);
    m(1, 1)      = 2.0 / (top - bottom);
    m(1, 3)      = -(top + bottom) / (top - bottom);
    m(2, 2)      = 1.0 / (farz - nearz);
    m(2, 3)      = farz / (farz - nearz);
    return m;
}

Matrix4d inline Perspective(double fov, double aspect, double nearz, double farz) noexcept {
    // Vertical field of view; clip z = (nearz * view_z + nearz * farz) / (farz - nearz) with
    // clip w = -view_z, so depth d in front of the eye yields nearz * (farz - d) / ((farz - nearz) * d).
    const double height = 1.0 / std::tan(fov / 2.0);
    const double width  = height / aspect;
    Matrix4d     m      = Matrix4d::Zero();
    m(0, 0)             = width;
    m(1, 1)             = height;
    m(2, 2)             = nearz / (farz - nearz);
    m(2, 3)             = nearz * farz / (farz - nearz);
    m(3, 2)             = -1.0;
    return m;
}
} // namespace Eigen
