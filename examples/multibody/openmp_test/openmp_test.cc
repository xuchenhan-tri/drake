#include <memory>

#include <gflags/gflags.h>

#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"

DEFINE_int64(num_iterations, 10000,
             "number of iterations to perform data update.");
DEFINE_double(resolution_hint, 1.0, "Resolution hint of the sphere.");
DEFINE_double(time_step, 5.0e-3,
              "Discrete time step for the system [s]. Must be positive.");

using drake::geometry::GeometryInstance;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::fem::DeformableBodyConfig;
using drake::multibody::fem::FemModel;
using drake::multibody::fem::FemState;
using drake::multibody::internal::DeformableBodyId;
using drake::multibody::internal::DeformableModel;
using drake::systems::Context;

namespace drake {
namespace examples {
namespace multibody {
namespace openmp_test {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;
  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";
  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Set up a deformable sphere. */
  auto deformable_model = std::make_unique<DeformableModel<double>>(&plant);

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(5e3);
  deformable_config.set_poissons_ratio(0.4);
  deformable_config.set_mass_density(1e3);
  deformable_config.set_stiffness_damping_coefficient(0.01);

  auto sphere = std::make_unique<Sphere>(1.0);
  const RigidTransformd X_WB(Vector3<double>(0.0, 0.0, 0.0));
  auto sphere_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(sphere), "sphere");
  const DeformableBodyId body_id = deformable_model->RegisterDeformableBody(
      std::move(sphere_instance), deformable_config, FLAGS_resolution_hint);
  const FemModel<double>& fem_model = deformable_model->GetFemModel(body_id);
  auto fem_state = fem_model.MakeFemState();
  fem_state->DisableCaching();
  for (int i = 0; i < FLAGS_num_iterations; ++i) {
    fem_model.ComputeData(*fem_state);
  }
  return 0;
}

}  // namespace
}  // namespace openmp_test
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Test openmp parallelization for FEM element data update");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::openmp_test::do_main();
}
