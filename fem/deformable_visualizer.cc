#include "drake/fem/deformable_visualizer.h"

#include <algorithm>
#include <utility>

#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_deformable_tri.hpp"
#include "drake/lcmt_deformable_tri_mesh_init.hpp"
#include "drake/lcmt_deformable_tri_mesh_update.hpp"

namespace drake {
namespace fem {

using std::vector;
using systems::Context;
using systems::EventStatus;

DeformableVisualizer::DeformableVisualizer(double update_period,
                                           std::string mesh_name,
                                           const vector<std::unique_ptr<FemTetMeshBase>>& meshes,
                                           lcm::DrakeLcmInterface* lcm)
    : lcm_(lcm), mesh_name_(std::move(mesh_name)) {
  if (lcm == nullptr) {
    owned_lcm_ = std::make_unique<lcm::DrakeLcm>();
    lcm_ = owned_lcm_.get();
  }
  Flatten(meshes);
  this->DeclareInputPort("vertex_positions", systems::kVectorValued,
                         volume_vertex_count_ * 3);
  this->DeclareInitializationPublishEvent(
      &DeformableVisualizer::PublishMeshInit);
  this->DeclarePeriodicPublishEvent(update_period, 0.0,
                                    &DeformableVisualizer::PublishMeshUpdate);
}

void DeformableVisualizer::SendMeshInit() const {
  lcmt_deformable_tri_mesh_init message;

  message.mesh_name = mesh_name_;
  message.num_vertices = static_cast<int>(surface_to_volume_vertices_.size());
  message.num_tris = static_cast<int>(surface_triangles_.size());
  message.tris.resize(message.num_tris);
  for (int t = 0; t < message.num_tris; ++t) {
    message.tris[t].vertices[0] = surface_triangles_[t](0);
    message.tris[t].vertices[1] = surface_triangles_[t](1);
    message.tris[t].vertices[2] = surface_triangles_[t](2);
  }

  lcm::Publish(lcm_, "DEFORMABLE_MESH_INIT", message);
}

EventStatus DeformableVisualizer::PublishMeshInit(
    const Context<double>&) const {
  SendMeshInit();
  return EventStatus::Succeeded();
}

void DeformableVisualizer::PublishMeshUpdate(
    const Context<double>& context) const {
  lcmt_deformable_tri_mesh_update message;
  message.timestamp =
      static_cast<int64_t>(ExtractDoubleOrThrow(context.get_time()) * 1e6);
  message.mesh_name = mesh_name_;
  const int v_count = static_cast<int>(surface_to_volume_vertices_.size());
  message.data_size = v_count * 3;
  message.data.resize(message.data_size);

  // The volume vertex positions are one flat vector. Such that the position of
  // the ith volume vertex is in entries j, j + 1, and j + 2, j = 3i.
  const auto& vertex_state = vertex_positions_input_port()
                                 .Eval<systems::BasicVector<double>>(context)
                                 .get_value();
  for (int v = 0; v < v_count; ++v) {
    const int out_index = v * 3;
    const int state_index = surface_to_volume_vertices_[v] * 3;
    message.data[out_index] = vertex_state(state_index);
    message.data[out_index + 1] = vertex_state(state_index + 1);
    message.data[out_index + 2] = vertex_state(state_index + 2);
  }
  lcm::Publish(lcm_, "DEFORMABLE_MESH_UPDATE", message);
}

}  // namespace fem
}  // namespace drake
