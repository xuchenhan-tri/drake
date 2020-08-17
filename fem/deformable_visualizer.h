#pragma once

#include <memory>
#include <string>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/lcm/drake_lcm_interface.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace fem {

/** A class for visualizing a deformable mesh in `drake_visualizer`.
 Specifically, it dispatches LCM messages to initialize a mesh definition and
 then updates vertex positions at a fixed frequency. Although the input is a
 volume (aka tetrahedral mesh), only the surface triangular mesh gets
 visualized.

 @system
 name: DeformableVisualizer
 input_ports:
 - vertex_positions
 @endsystem

 The input port is a vector-valued port containing 3N values, where N + 1 is
 the *largest* vertex index referenced in the tetrahedral mesh topology. For the
 iᵗʰ vertex, its x-, y-, and z-positions (measured and expressed in the *world*
 frame) are the 3i, 3i + 1, and 3i + 2 elements of the input port.  */
class DeformableVisualizer : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DeformableVisualizer)

  /** Constructs a visualizer for a single tetrahedral mesh. The mesh topology
   is given by a collection of tets (each represented by 4 indices into an
   *implied* ordered set of vertex positions).

   @param update_period  The duration (in seconds) between publications of mesh
                         state.
   @param mesh_name      The name for the mesh (as it will appear in
                         drake_visualizer).
   @param tet_mesh       The definition of the tetrahedral mesh topology.
   @param lcm            If non-nullptr, `this` will use the provided lcm
                         interface to broadcast lcm messages. The system will
                         keep a reference to the instance and it should stay
                         alive as long as this class. Otherwise, `this` will
                         create its own instance.
   @pre update_period > 0.  */
  DeformableVisualizer(double update_period, std::string mesh_name,
                       const std::vector<Vector4<int>>& tet_mesh,
                       lcm::DrakeLcmInterface* lcm = nullptr);

  /** Send the mesh initialization message. This can be invoked explicitly but
   is generally not necessary. The initialization method is also called by
   an initialization event.  */
  void SendMeshInit() const;

  /** Returns the input port for vertex positions.  */
  const systems::InputPort<double>& vertex_positions_input_port() const {
    return systems::System<double>::get_input_port(0);
  }

 private:
  /* Analyzes the tet mesh topology to do the following:
    1. Build a surface mesh from the volume mesh.
    2. Create a mapping from surface vertex to volume vertex.
    3. Record the expected number of vertices referenced by the tet mesh.  */
  void AnalyzeTets(const std::vector<Vector4<int>>& tet_mesh);

  /* Call SendMeshInit in an initialization event. */
  systems::EventStatus PublishMeshInit(const systems::Context<double>&) const;

  /* Broadcast the current vertex positions for the mesh.  */
  void PublishMeshUpdate(const systems::Context<double>& context) const;

  /* The LCM interface used to broadcast its messages. This system can
   optionally own its lcm interface.  */
  lcm::DrakeLcmInterface* lcm_{};
  std::unique_ptr<lcm::DrakeLcmInterface> owned_lcm_{};

  /* The name of the mesh. Included in all lcm messages.  */
  const std::string mesh_name_;

  /* An *implicit* map from *surface* vertex indices to volume vertex indices.
   The iᵗʰ surface vertex corresponds to the volume vertex with index
   `surface_to_volume_vertices_[i]`.  */
  std::vector<int> surface_to_volume_vertices_;

  /* A surface mesh representing the topology of the volume mesh's surface.  */
  std::vector<Vector3<int>> surface_triangles_;

  // The total number of vertices expected on the input port as implied by the
  // tetrahedra definitions.
  int volume_vertex_count_{};
};
}  // namespace fem
}  // namespace drake
