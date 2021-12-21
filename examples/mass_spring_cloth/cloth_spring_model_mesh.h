#pragma once
#include <vector>

#include "drake/examples/mass_spring_cloth/cloth_spring_model.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace examples {
namespace mass_spring_cloth {

class ClothSpringModelMesh final : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ClothSpringModelMesh);

  static const ClothSpringModelMesh& AddToBuilder(
      systems::DiagramBuilder<double>* builder,
      geometry::MeshcatVisualizerd* meshcat_visualizer,
      const ClothSpringModel<double>& cloth_spring_model);

 private:
  ClothSpringModelMesh(int nx, int ny);
  void OutputMesh(const systems::Context<double>&,
                  geometry::TriangleSurfaceMesh<double>*) const;
  int nx_{0};
  int ny_{0};
  std::vector<drake::geometry::SurfaceTriangle> elements_;
};

}  // namespace mass_spring_cloth
}  // namespace examples
}  // namespace drake
