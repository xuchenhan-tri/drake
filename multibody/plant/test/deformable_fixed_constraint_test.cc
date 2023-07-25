#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/deformable_driver.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

using drake::geometry::Box;
using drake::geometry::GeometryId;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::ProximityProperties;
using drake::geometry::SceneGraph;
using drake::geometry::SceneGraphInspector;
using drake::geometry::Sphere;
using drake::geometry::VolumeMesh;
using drake::math::RigidTransformd;
using drake::multibody::fem::FemState;
using drake::systems::Context;
using drake::systems::Simulator;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using std::make_unique;
using std::unique_ptr;

namespace drake {
namespace multibody {
namespace internal {

/* Deformable body parameters.  */
constexpr double kYoungsModulus = 1e4;      // unit: N/m²
constexpr double kPoissonsRatio = 0.45;     // unitless.
constexpr double kMassDensity = 1e3;        // unit: kg/m³
constexpr double kStiffnessDamping = 0.01;  // unit: s
/* Time step (seconds). */
constexpr double kDt = 1e-2;
const CoulombFriction<double> kFriction{0.4, 0.4};

/* Sets up a deformable simulation with a deformable octahedron fixed to a
 static rigid box on the top and to a dynamic rigid box on the bottom.

                             ----------------
                             |              |
                             |              |
                             | static rigid |
                             |              |
                             |              |
                             -------/|\------
                                   / | \
                                  /  |  \
                                 /___|___\  deformable
                                 \   |   /
                                  \  |  /
                                   \ | /
                             -------\|/------
                             |              |
                             |              |
                             | dynamic rigid|
                             |              |
                             |              |
                             ----------------

 The setup is used to test the discrete updates for deformable bodies subject to
 fixed constraint in a simulator. Run:

 bazel run //tools:meldis -- --open-window
 bazel-bin/multibody/plant/deformable_fixed_constraint_test

 to visualize the test. */
class DeformableFixedConstraintTest : public ::testing::Test {
 protected:
  /* Sets up a scene with a deformable octahedron fixed to a static rigid box on
   one end and to a dynamic rigid box on the other end. */
  void SetUp() override {
    systems::DiagramBuilder<double> builder;
    MultibodyPlantConfig plant_config;
    plant_config.time_step = kDt;
    plant_config.discrete_contact_solver = "sap";
    std::tie(plant_, scene_graph_) = AddMultibodyPlant(plant_config, &builder);

    auto deformable_model = make_unique<DeformableModel<double>>(plant_);
    deformable_body_id_ = RegisterDeformableOctahedron(
        deformable_model.get(), RigidTransformd(Vector3d(0, 0, -0.1)), 0.1,
        "deformable");
    model_ = deformable_model.get();
    plant_->AddPhysicalModel(std::move(deformable_model));

    rigid_body_index_ =
        plant_
            ->AddRigidBody("rigid_box",
                           SpatialInertia<double>::SolidCubeWithMass(1.0, 0.1))
            .index();
    ProximityProperties proximity_prop;
    geometry::AddContactMaterial({}, {}, kFriction, &proximity_prop);
    geometry::AddCompliantHydroelasticProperties(1.0, 1e6, &proximity_prop);
    IllustrationProperties illustration_props;
    illustration_props.AddProperty("phong", "diffuse",
                                   Vector4d(0.7, 0.5, 0.4, 0.8));
    const Box box(0.2, 0.2, 0.2);
    const auto& box_body = plant_->get_body(rigid_body_index_);
    plant_->RegisterCollisionGeometry(box_body, RigidTransformd::Identity(),
                                      box, "box_collision", proximity_prop);
    plant_->RegisterVisualGeometry(box_body, RigidTransformd::Identity(), box,
                                   "box_visual", illustration_props);

    /* Initialize the box to the bottom of the deformable octahedron so that the
     bottom (with smallest z coordinate in the world frame) vertex is fixed to
     the box. */
    X_WB_ = RigidTransformd(Vector3d(0, 0, -0.29));
    model_->AddFixedConstraint(deformable_body_id_, box_body, X_WB_, box,
                               RigidTransformd::Identity());

    /* Initialize the static box to the top of the deformable octahedron so that
     the top (with largest z coordinate in the world frame) vertex is fixed to
     the static box. */
    const Box static_box(0.2, 0.2, 0.2);
    const RigidTransformd X_WG(Vector3d(0, 0, 0.09));
    plant_->RegisterCollisionGeometry(plant_->world_body(), X_WG, static_box,
                                      "static_box_collision", proximity_prop);
    plant_->RegisterVisualGeometry(plant_->world_body(), X_WG, static_box,
                                   "static_box_visual", illustration_props);
    model_->AddFixedConstraint(deformable_body_id_, plant_->world_body(),
                               RigidTransformd::Identity(), static_box, X_WG);
    plant_->Finalize();

    /* Connect visualizer. Useful for when this test is used for debugging. */
    geometry::DrakeVisualizerd::AddToBuilder(&builder, *scene_graph_);

    builder.Connect(model_->vertex_positions_port(),
                    scene_graph_->get_source_configuration_port(
                        plant_->get_source_id().value()));

    diagram_ = builder.Build();
  }

  SceneGraph<double>* scene_graph_{nullptr};
  MultibodyPlant<double>* plant_{nullptr};
  DeformableModel<double>* model_{nullptr};
  const CompliantContactManager<double>* manager_{nullptr};
  unique_ptr<systems::Diagram<double>> diagram_{nullptr};
  BodyIndex rigid_body_index_;
  RigidTransformd X_WB_;  // Initial pose of the dynamic rigid body.
  DeformableBodyId deformable_body_id_;

 private:
  /* Constructs a deformable sphere with the given `radius` and pose in the
   world frame (X_WD) discretized as an octrahedron volume mesh with 7 vertices
   and 8 tetrahedra and registers the body with the given `model`. */
  DeformableBodyId RegisterDeformableOctahedron(DeformableModel<double>* model,
                                                const RigidTransformd& X_WD,
                                                double radius,
                                                std::string name) {
    auto geometry = make_unique<GeometryInstance>(
        X_WD, make_unique<Sphere>(radius), std::move(name));
    ProximityProperties props;
    geometry::AddContactMaterial({}, {}, kFriction, &props);
    geometry->set_proximity_properties(std::move(props));
    fem::DeformableBodyConfig<double> body_config;
    body_config.set_youngs_modulus(kYoungsModulus);
    body_config.set_poissons_ratio(kPoissonsRatio);
    body_config.set_mass_density(kMassDensity);
    body_config.set_stiffness_damping_coefficient(kStiffnessDamping);
    /* Make the resolution hint very large so that we get an octahedron. */
    const double kRezHint = 10.0 * radius;
    DeformableBodyId id = model->RegisterDeformableBody(std::move(geometry),
                                                        body_config, kRezHint);
    /* Verify that the geometry has 7 vertices and is indeed an octahedron. */
    const SceneGraphInspector<double>& inspector =
        scene_graph_->model_inspector();
    GeometryId g_id = model->GetGeometryId(id);
    const VolumeMesh<double>* mesh_G = inspector.GetReferenceMesh(g_id);
    DRAKE_DEMAND(mesh_G != nullptr);
    DRAKE_DEMAND(mesh_G->num_vertices() == 7);
    return id;
  }
};

namespace {

TEST_F(DeformableFixedConstraintTest, SteadyState) {
  Simulator<double> simulator(*diagram_);
  Context<double>& mutable_diagram_context = simulator.get_mutable_context();
  Context<double>& mutable_plant_context =
      plant_->GetMyMutableContextFromRoot(&mutable_diagram_context);
  plant_->SetFreeBodyPose(&mutable_plant_context,
                          plant_->get_body(rigid_body_index_), X_WB_);

  /* Run simulation for long enough to reach steady state. */
  simulator.AdvanceTo(10.0);

  /* Verify the system has reached steady state. */
  const Context<double>& diagram_context = simulator.get_context();
  const Context<double>& plant_context =
      plant_->GetMyContextFromRoot(diagram_context);
  VectorXd discrete_state =
      plant_context
          .get_discrete_state(
              model_->GetDiscreteStateIndex(deformable_body_id_))
          .value();
  const int num_nodes = model_->GetFemModel(deformable_body_id_).num_nodes();
  const VectorXd q = discrete_state.head(3 * num_nodes);
  const VectorXd v = discrete_state.segment(3 * num_nodes, 3 * num_nodes);
  constexpr double kVelocityThreshold = 1e-6;      // unit: m/s.
  EXPECT_TRUE(CompareMatrices(v, VectorXd::Zero(v.size()), kVelocityThreshold));
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
