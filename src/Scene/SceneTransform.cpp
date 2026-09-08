#include "SceneTransform.h"

using namespace wallpaper;
using namespace Eigen;

Matrix4d SceneTransform::LocalMatrix() const {
    Affine3d transform = Affine3d::Identity();
    transform.prescale(m_scale.cast<double>());
    transform.prerotate(AngleAxis<double>(m_rotation.x(), Vector3d::UnitX()));
    transform.prerotate(AngleAxis<double>(m_rotation.y(), Vector3d::UnitY()));
    transform.prerotate(AngleAxis<double>(m_rotation.z(), Vector3d::UnitZ()));
    transform.pretranslate(m_translate.cast<double>());
    transform.translate(m_alignment_offset.cast<double>());
    return transform.matrix();
}

void SceneTransform::SetLocalAffine(const Affine3f& affine) {
    // Parent/attachment rebinding supplies the complete local matrix, including card alignment.
    // Remove that local offset before decomposition so it is not baked into the script origin
    // and applied a second time by LocalMatrix(). Preserve the established Z/Y/X Euler order.
    Affine3f authored = affine;
    authored.translate(-m_alignment_offset);
    Matrix3f linear = authored.linear();
    Vector3f scale(linear.col(0).norm(), linear.col(1).norm(), linear.col(2).norm());
    for (int axis = 0; axis < 3; ++axis) {
        if (scale[axis] > 1e-6f) {
            linear.col(axis) /= scale[axis];
        } else {
            // A collapsed axis retains authored zero scale. Supply only its missing direction to
            // the Euler decomposition, so scripts hiding a layer with scale zero keep it collapsed.
            linear.col(axis).setZero();
            linear(axis, axis) = 1.0f;
            scale[axis] = 0.0f;
        }
    }
    const auto zyx = linear.eulerAngles(2, 1, 0);
    m_scale = scale;
    m_rotation = Vector3f(zyx[2], zyx[1], zyx[0]);
    m_translate = authored.translation();
    ++m_revision;
}

void SceneTransform::CopyFrom(const SceneTransform& transform) {
    m_translate = transform.m_translate;
    m_scale = transform.m_scale;
    m_rotation = transform.m_rotation;
    m_alignment_offset = transform.m_alignment_offset;
    ++m_revision;
}
