#pragma once

#include <cstdint>
#include <Eigen/Geometry>
#include "Core/NoCopyMove.hpp"

namespace wallpaper
{

// Live placement belongs to a scene object independently from any render-node lifetime. Draw
// handles may share this record with their owner, while private camera-local phases have their
// own local record. Revision changes let matrix consumers observe writes made directly through
// SceneObject without requiring a SceneNode setter to invalidate a physical child subtree.
class SceneTransform : NoCopy, NoMove {
public:
    SceneTransform() = default;
    SceneTransform(const Eigen::Vector3f& translate, const Eigen::Vector3f& scale,
                   const Eigen::Vector3f& rotation)
        : m_translate(translate), m_scale(scale), m_rotation(rotation) {}

    const Eigen::Vector3f& Translate() const { return m_translate; }
    const Eigen::Vector3f& Scale() const { return m_scale; }
    const Eigen::Vector3f& Rotation() const { return m_rotation; }
    const Eigen::Vector3f& AlignmentOffset() const { return m_alignment_offset; }
    uint64_t Revision() const { return m_revision; }

    void SetTranslate(const Eigen::Vector3f& value) { m_translate = value; ++m_revision; }
    void SetScale(const Eigen::Vector3f& value) { m_scale = value; ++m_revision; }
    void SetRotation(const Eigen::Vector3f& value) { m_rotation = value; ++m_revision; }
    void SetAlignmentOffset(const Eigen::Vector3f& value) {
        m_alignment_offset = value;
        ++m_revision;
    }

    Eigen::Matrix4d LocalMatrix() const;
    void SetLocalAffine(const Eigen::Affine3f& affine);
    void CopyFrom(const SceneTransform& transform);

private:
    Eigen::Vector3f m_translate { 0.0f, 0.0f, 0.0f };
    Eigen::Vector3f m_scale { 1.0f, 1.0f, 1.0f };
    Eigen::Vector3f m_rotation { 0.0f, 0.0f, 0.0f };
    // Alignment moves the local card around the authored pivot; it is not part of the origin
    // exposed to scripts. It therefore follows rotation/scale when forming the local matrix.
    Eigen::Vector3f m_alignment_offset { 0.0f, 0.0f, 0.0f };
    uint64_t m_revision { 1 };
};

} // namespace wallpaper
