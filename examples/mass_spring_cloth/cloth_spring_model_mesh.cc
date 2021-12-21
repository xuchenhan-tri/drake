#include "drake/examples/mass_spring_cloth/cloth_spring_model_mesh.h"

#include <memory>
#include <utility>

#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace examples {
namespace mass_spring_cloth {

using Eigen::Vector3d;
using Eigen::Vector4d;
const ClothSpringModelMesh& ClothSpringModelMesh::AddToBuilder(
    systems::DiagramBuilder<double>* builder,
    geometry::MeshcatVisualizerd* meshcat_visualizer,
    const ClothSpringModel<double>& cloth_spring_model) {
  DRAKE_THROW_UNLESS(builder != nullptr);
  DRAKE_THROW_UNLESS(meshcat_visualizer != nullptr);

  auto cloth_spring_model_geometry = builder->AddSystem(
      std::unique_ptr<ClothSpringModelMesh>(new ClothSpringModelMesh(
          cloth_spring_model.nx(), cloth_spring_model.ny())));
  builder->Connect(cloth_spring_model.get_output_port(0),
                   cloth_spring_model_geometry->get_input_port(0));
  builder->Connect(cloth_spring_model_geometry->get_output_port(0),
                   meshcat_visualizer->mesh_input_port());

  return *cloth_spring_model_geometry;
}

ClothSpringModelMesh::ClothSpringModelMesh(int nx, int ny) : nx_(nx), ny_(ny) {
  this->DeclareInputPort("particle_positions", systems::kVectorValued,
                         nx * ny * 3);
  this->DeclareAbstractOutputPort("mesh", &ClothSpringModelMesh::OutputMesh);
  elements_.clear();
  for (int i = 0; i < nx - 1; ++i) {
    for (int j = 0; j < ny - 1; ++j) {
      int current = i * ny + j;
      elements_.emplace_back(current, current + 1, current + ny);
      elements_.emplace_back(current + ny, current + 1, current + ny + 1);
    }
  }
}

void ClothSpringModelMesh::OutputMesh(
    const systems::Context<double>& context,
    drake::geometry::TriangleSurfaceMesh<double>* mesh) const {
  const auto& input = get_input_port(0).Eval(context);
  std::vector<Vector3d> positions;
  positions.reserve(nx_ * ny_);
  for (int i = 0; i < nx_ * ny_; ++i) {
    const double x = input(3 * i);
    const double y = input(3 * i + 1);
    const double z = input(3 * i + 2);
    positions.emplace_back(x, y, z);
  }
  auto elements = elements_;
  *mesh = drake::geometry::TriangleSurfaceMesh(std::move(elements),
                                              std::move(positions));
}

}  // namespace mass_spring_cloth
}  // namespace examples
}  // namespace drake
