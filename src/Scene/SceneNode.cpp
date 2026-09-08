#include "SceneNode.h"

#include <Eigen/Geometry>

using namespace wallpaper;
using namespace Eigen;

void SceneNode::UpdateTrans() {
    // Always resolve the parent before comparing revisions. Object state can be edited without
    // going through this draw handle, and a clean child must still observe such an ancestor edit.
    // Physical reparenting sets m_dirty; ordinary TRS/alignment changes advance the shared local
    // record. Each rebuilt model advances its own revision to carry either change down the tree.
    if (m_parent != nullptr) m_parent->UpdateTrans();
    const auto parent_revision = m_parent != nullptr ? m_parent->m_model_revision : 0;
    const auto local_revision = m_transform->Revision();
    if (!m_dirty && m_local_revision == local_revision &&
        m_parent_revision == parent_revision) return;

    m_trans = m_parent != nullptr
        ? (m_parent->ModelTrans() * m_transform->LocalMatrix()).eval()
        : m_transform->LocalMatrix();
    m_local_revision = local_revision;
    m_parent_revision = parent_revision;
    m_dirty = false;
    ++m_model_revision;
}
