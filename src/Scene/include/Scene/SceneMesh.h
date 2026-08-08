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
