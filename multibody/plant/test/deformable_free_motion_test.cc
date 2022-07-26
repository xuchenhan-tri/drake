#include <gtest/gtest.h>

#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/multibody/plant/compliant_contact_manager.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace internal {
namespace {

using fem::DeformableBodyConfig;
using geometry::GeometryInstance;
using geometry::SceneGraph;
using geometry::Sphere;
using math::RigidTransformd;
using std::make_unique;

class DeformableFreeMotionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    constexpr double kDt = 1e-3;
    std::tie(plant_, scene_graph_) =
        AddMultibodyPlantSceneGraph(&builder_, kDt);
    auto deformable_model = make_unique<DeformableModel<double>>(plant_);
    deformable_model_ptr_ = deformable_model.get();
    plant_->AddPhysicalModel(move(deformable_model));
  }

  systems::DiagramBuilder<double> builder_;
  DeformableModel<double>* deformable_model_ptr_{nullptr};
  MultibodyPlant<double>* plant_{nullptr};
  SceneGraph<double>* scene_graph_{nullptr};
};

TEST_F(DeformableFreeMotionTest, RegisterDeformableBody) {
  constexpr double kRezHint = 0.5;
  auto geometry = make_unique<GeometryInstance>(
      RigidTransformd(), make_unique<Sphere>(1), "sphere");
  deformable_model_ptr_->RegisterDeformableBody(
      std::move(geometry), DeformableBodyConfig<double>(), kRezHint);
  EXPECT_EQ(deformable_model_ptr_->num_bodies(), 1);
  plant_->Finalize();
}

}  // namespace
}  // namespace internal
}  // namespace multibody
}  // namespace drake
