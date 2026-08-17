#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <atomic>
#include <string>
#include <Eigen/Geometry>

#include "SceneVertexArray.h"
#include "SceneIndexArray.h"
#include "SceneMaterial.h"

namespace wallpaper
{
class SceneMesh {
public:
	struct DrawRange {
		uint32_t firstIndex { 0 };
		uint32_t indexCount { 0 };
	};
	struct MaskedDrawGroup {
		std::string maskTexture;
		std::vector<DrawRange> maskRanges;
		std::vector<DrawRange> contentRanges;
	};
	struct MaskedDrawRange {
		DrawRange range;
		int32_t groupIndex { -1 };
	};
	struct MaskedDrawPlan {
		std::vector<DrawRange> unmaskedRanges;
		std::vector<MaskedDrawGroup> groups;
		std::vector<MaskedDrawRange> orderedRanges;

		bool empty() const { return groups.empty(); }
	};
	struct SkinningInfo {
		uint32_t boneCount { 0 };
	};

	SceneMesh(bool dynamic = false):m_dynamic(dynamic),m_dirty(false),
		m_data(std::make_shared<Data>()) {}

	std::size_t VertexCount() const { return m_data->vertexArrays.size(); }
	std::size_t IndexCount() const { return m_data->indexArrays.size(); }

	MeshPrimitive Primitive() const { return m_primitive; }
	uint32_t PointSize() const { return m_pointSize; }

	bool Dynamic() const { return m_dynamic; }
	const auto& Dirty() const { return m_dirty; }
	auto& Dirty() { return m_dirty; }
	void SetDirty() { m_dirty.store(true); }

	uint32_t ID() const { return m_id; };
	void SetID(uint32_t v) { m_id = v; };

	const SceneVertexArray& GetVertexArray(const std::size_t index) const { return m_data->vertexArrays[index]; }
	const SceneIndexArray& GetIndexArray(const std::size_t index) const { return m_data->indexArrays[index]; }	
	const MaskedDrawPlan& MaskedDraw() const { return m_data->maskedDraw; }
	const SkinningInfo& Skinning() const { return m_data->skinning; }
	// Geometry transforms are the mesh-side coordinate contract. They are applied by the uniform
	// updater to every matrix path (including masked ranges) without mutating imported vertex bytes.
	const Eigen::Affine3f& GeometryTransform() const { return m_data->geometry_transform; }
	void SetGeometryTransform(const Eigen::Affine3f& transform) {
		m_data->geometry_transform = transform;
		SetDirty();
	}

	SceneVertexArray& GetVertexArray(const std::size_t index) { return m_data->vertexArrays[index]; }
	SceneIndexArray& GetIndexArray(const std::size_t index) { return m_data->indexArrays[index]; }	


	void AddIndexArray(SceneIndexArray&& array) {
		m_data->indexArrays.emplace_back(std::move(array));
	}
	uint32_t IndexElementBytes() const { return m_data->index_element_bytes; }
	void SetIndexElementBytes(uint32_t bytes) {
		m_data->index_element_bytes = (bytes == 4) ? 4u : 2u;
	}
	uint32_t LogicalIndexCount() const {
		if (m_data->indexArrays.empty()) return 0;
		const auto& indice = m_data->indexArrays[0];
		if (m_data->index_element_bytes == 4) {
			return static_cast<uint32_t>(indice.DataCount());
		}
		const size_t packed = (indice.DataCount() * 2) / 3;
		return static_cast<uint32_t>(packed * 3);
	}
	void AddVertexArray(SceneVertexArray&& array) {
		m_data->vertexArrays.emplace_back(std::move(array));
	}
	void SetMaskedDraw(MaskedDrawPlan&& plan) {
		m_data->maskedDraw = std::move(plan);
	}
	void SetSkinning(SkinningInfo info) { m_data->skinning = info; }
	void AddMaterial(SceneMaterial&& material) {
		m_material = std::make_shared<SceneMaterial>(material);
	}

	void SetPrimitive(MeshPrimitive v) {  m_primitive = v; }
	void SetPointSize(uint32_t v) { m_pointSize = v; }

	// File mdl chunks own exact-size GPU VB/IB pairs. Generated image cards stay on the
	// shared small static pool because they are authored at runtime and can be resized.
	bool FileImmutable() const { return m_data->file_immutable; }
	void SetFileImmutable(bool v) { m_data->file_immutable = v; }
	const void* GpuStorageKey() const { return m_data.get(); }

	bool HasBounds() const { return m_data->bounds_valid; }
	const Eigen::Vector3f& BoundsMin() const { return m_data->bounds_min; }
	const Eigen::Vector3f& BoundsMax() const { return m_data->bounds_max; }
	void SetBounds(const Eigen::Vector3f& min, const Eigen::Vector3f& max) {
		m_data->bounds_min   = min;
		m_data->bounds_max   = max;
		m_data->bounds_valid = min.allFinite() && max.allFinite() &&
		                       (min.array() <= max.array()).all();
	}

	bool HasCpuPayload() const {
		return VertexCount() > 0 && GetVertexArray(0).Data() != nullptr;
	}
	std::size_t ReleaseCpuPayload() {
		std::size_t bytes = 0;
		for (auto& vertex : m_data->vertexArrays) bytes += vertex.ReleaseCpuPayload();
		for (auto& index : m_data->indexArrays) bytes += index.ReleaseCpuPayload();
		return bytes;
	}

	SceneMaterial* Material() { return m_material.get(); }

	void ChangeMeshDataFrom(const SceneMesh& o) {
		m_data = o.m_data;
	}

private:
	struct Data {
		std::vector<SceneVertexArray> vertexArrays;
		std::vector<SceneIndexArray> indexArrays;
		MaskedDrawPlan maskedDraw;
		SkinningInfo skinning;
		Eigen::Affine3f geometry_transform { Eigen::Affine3f::Identity() };
		Eigen::Vector3f bounds_min { Eigen::Vector3f::Zero() };
		Eigen::Vector3f bounds_max { Eigen::Vector3f::Zero() };
		uint32_t index_element_bytes { 2 };
		bool file_immutable { false };
		bool bounds_valid { false };
	};

	uint32_t m_id { std::numeric_limits<uint32_t>::max() };
	MeshPrimitive m_primitive {MeshPrimitive::TRIANGLE};
	uint32_t m_pointSize {1};
	bool m_dynamic;
	std::atomic<bool> m_dirty;

	std::shared_ptr<Data> m_data;
	std::shared_ptr<SceneMaterial> m_material;
};

}
