#include <gflags/gflags.h>

#include "drake/geometry/proximity/make_cylinder_mesh.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"

namespace drake {
namespace examples {

DEFINE_double(L, 1.0, "Height of the cylinder");
DEFINE_double(R, 1.0, "Radius of the cylinder");
DEFINE_double(resolution, 0.5, "Resolution hint");

using geometry::VolumeMesh;

int do_main() {
  auto mesh = geometry::internal::MakeCylinderVolumeMesh<double>(
      geometry::Cylinder(FLAGS_R, FLAGS_L), FLAGS_resolution);
  geometry::internal::WriteVolumeMeshToVtk("cylinder_mesh.vtk", mesh,
                                           "Cylinder");
  return 0;
}

}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::do_main();
}
