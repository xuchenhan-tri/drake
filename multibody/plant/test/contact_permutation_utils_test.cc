#include "drake/multibody/plant/contact_permutation_utils.h"

#include <limits>

#include <gtest/gtest.h>

//#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include <iostream>

#include "drake/common/drake_assert.h"
#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

#define PRINT_VEC(v)       \
  std::cout << #v ":\n";   \
  for (auto& e : v) {      \
    std::cout << e << " "; \
  }                        \
  std::cout << std::endl;

using drake::systems::Context;

using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RotationMatrix;
using drake::math::RotationMatrixd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using PermutationMatrixXd =
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
using drake::multibody::contact_solvers::internal::BlockSparseMatrix;

namespace drake {
namespace multibody {

// TODO(amcastro-tri): this test comes from multibody_plant_test.cc
// StateSelection__FloatingBodies. Refactor to reuse this code.
void AddScenarioWithTwoArms(MultibodyPlant<double>* plant) {
  DRAKE_DEMAND(!plant->is_finalized());

  const std::string iiwa_sdf_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/sdf/"
      "iiwa14_no_collision.sdf");

  const std::string table_sdf_path = FindResourceOrThrow(
      "drake/examples/kuka_iiwa_arm/models/table/"
      "extra_heavy_duty_table_surface_only_collision.sdf");

  const std::string mug_sdf_path =
      FindResourceOrThrow("drake/examples/simple_gripper/simple_mug.sdf");

  // Load a model of a table for the robot.
  Parser parser(&*plant);
  const ModelInstanceIndex robot_table_model =
      parser.AddModelFromFile(table_sdf_path, "robot_table");
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("link", robot_table_model));

  // Load the robot and weld it on top of the robot table.
  const ModelInstanceIndex arm1_model =
      parser.AddModelFromFile(iiwa_sdf_path, "robot1");

  // Add a floating mug.
  // const ModelInstanceIndex mug_model = ...
  parser.AddModelFromFile(mug_sdf_path, "mug1");
  // const Body<double>& mug = plant->GetBodyByName("main_body", mug_model);

  const ModelInstanceIndex arm2_model =
      parser.AddModelFromFile(iiwa_sdf_path, "robot2");

  // Add a second floating mug.
  parser.AddModelFromFile(mug_sdf_path, "mug2");

  const double table_top_z_in_world =
      // table's top height
      0.736 +
      // table's top width
      0.057 / 2;
  const RigidTransformd X_WLink0(Vector3d(0, 0, table_top_z_in_world));
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("iiwa_link_0", arm1_model), X_WLink0);

  // Second arm.
  const RigidTransformd X_WRobot2Link0(Vector3d(0.5, 0, table_top_z_in_world));
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("iiwa_link_0", arm2_model),
                    X_WRobot2Link0);

  // Load a second table for objects.
  const ModelInstanceIndex objects_table_model =
      parser.AddModelFromFile(table_sdf_path, "objects_table");
  const RigidTransformd X_WT(Vector3d(0.8, 0.0, 0.0));
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("link", objects_table_model), X_WT);

  // Define a fixed frame on the -x, -y corner of the objects table.
  // const RigidTransformd X_TO(RotationMatrixd::MakeXRotation(-M_PI_2),
  //                         Vector3d(-0.3, -0.3, table_top_z_in_world));
  // const auto& objects_frame_O =
  //    plant->AddFrame(std::make_unique<FixedOffsetFrame<double>>(
  //      "objects_frame", plant->GetFrameByName("link", objects_table_model),
  //    X_TO));

  plant->Finalize();
}

class MultibodyPlantTester {
 public:
  MultibodyPlantTester() = delete;

  static const internal::MultibodyTreeTopology& get_topology(
      const MultibodyPlant<double>& plant) {
    DRAKE_DEMAND(plant.is_finalized());
    return plant.internal_tree().get_topology();
  }

  static const MatrixXd& EvalContactJacobian(
      const MultibodyPlant<double>& plant,
      const systems::Context<double>& context) {
    return plant.EvalContactJacobians(context).Jc;
  }
};

namespace {

// Helper method to unit test that we can reconstruct the original mass matrix
// computed by MultibodyPlant from the per-tree block diagonal mass matrices.
void VerifyMassMatrixReconstructionFromDiagonalBlocks(
    const BlockSparseMatrix<double>& Mt,
    const std::vector<std::vector<int>>& velocity_permutation,
    const std::vector<int>& participating_trees, const MatrixXd& M) {
  int nv = 0;
  for (const auto& tree_permutation : velocity_permutation) {
    nv += tree_permutation.size();
  }

  // Negative value indicates no presence in Mt.
  std::vector<int> reduced_index(nv, -1);
  for (size_t tr = 0; tr < participating_trees.size(); ++tr) {
    const int t = participating_trees[tr];
    reduced_index[t] = tr;
  }

  // We'll build the permuted matrix first.
  MatrixXd M_reconstructed = MatrixXd::Zero(nv, nv);
  const int num_trees = velocity_permutation.size();
  for (int t = 0; t < num_trees; ++t) {
    const int nt = velocity_permutation[t].size();

    // Some trees t are not in contact and do not participate. Therefore there
    // is no corresponding block in Mt.
    const int tr = reduced_index[t];
    const MatrixXd* Mblock = tr >= 0 ? &Mt.get_block(tr) : nullptr;

    for (int it = 0; it < nt; ++it) {
      const int i = velocity_permutation[t][it];
      for (int jt = 0; jt < nt; ++jt) {
        const int j = velocity_permutation[t][jt];

        // When a tree does not participate, there is no block Mt for it.
        // For that case we just simply copy the block from the original matrix
        // so that the comparison of M and M_reconstructed is simpler.
        if (Mblock) {
          M_reconstructed(i, j) = (*Mblock)(it, jt);
        } else {
          M_reconstructed(i, j) = M(i, j);
        }
      }
    }
  }

#if 0
  // Build permutation matrix, from DFT to BFT (original).
  VectorX<int> perm_1d(nv);
  int v_1d = 0;
  for (const auto& tree_velocities : velocity_permutation) {
    for (int v : tree_velocities) {
      perm_1d[v_1d++] = v;
    }
  }
  const PermutationMatrixXd P_d2b(perm_1d);

  // Permute Mperm to get a reconstruction ot the original matrix.
  const MatrixXd M_reconstructed = P_d2b * Mperm * P_d2b.transpose();
#endif

  // Verify the result.
  EXPECT_TRUE(CompareMatrices(M, M_reconstructed,
                              std::numeric_limits<double>::epsilon()));
}

void VerifyJacobianReconstructionFromBlocks(
    const BlockSparseMatrix<double>& J_blocks,
    const internal::ContactGraph& graph,
    const std::vector<std::vector<int>>& velocity_permutation,
    const std::vector<int>& participating_trees, const MatrixXd& Jc) {
  int nv = 0;
  for (const auto& tree_permutation : velocity_permutation) {
    nv += tree_permutation.size();
  }
  ASSERT_EQ(Jc.cols(), nv);

  const auto& patches = graph.patches;
  int nc = 0;
  for (auto& p : patches) {
    nc += p.contacts.size();
  }
  ASSERT_EQ(Jc.rows(), 3 * nc);

  MatrixXd Jc_reconstructed(3 * nc, nv);
  Jc_reconstructed.setZero();

  for (const auto& [p, tr, Jpt] : J_blocks.get_blocks()) {
    const int t = participating_trees[tr];
    const int rp = patches[p].contacts.size();
    const int nt = velocity_permutation[t].size();

    EXPECT_TRUE((patches[p].t1 == tr) || (patches[p].t2 == tr));

    for (int kp = 0; kp < rp; ++kp) {
      const int k = patches[p].contacts[kp];
      for (int vt = 0; vt < nt; ++vt) {
        const int v = velocity_permutation[t][vt];
        Jc_reconstructed.block(3 * k, v, 3, 1) = Jpt.block(3 * kp, vt, 3, 1);
      }
    }
  }

  // Verify the result.
  EXPECT_TRUE(CompareMatrices(Jc, Jc_reconstructed,
                              std::numeric_limits<double>::epsilon()));
}

GTEST_TEST(DofsPermutation, VelocitiesPermutation) {
  MultibodyPlant<double> plant(0.0);
  AddScenarioWithTwoArms(&plant);
  const int nv = plant.num_velocities();

  std::vector<std::vector<int>> velocity_permutation;
  std::vector<int> body_to_tree_map;

  const internal::MultibodyTreeTopology& topology =
      MultibodyPlantTester::get_topology(plant);
  drake::multibody::internal::ComputeBfsToDfsPermutation(
      topology, &velocity_permutation, &body_to_tree_map);

  int nperm = 0;
  for (const auto& tree_permutation : velocity_permutation) {
    nperm += tree_permutation.size();
  }
  EXPECT_EQ(nperm, nv);

  // TODO: make these prints proper tests.
  PRINT_VAR(plant.num_velocities());
  PRINT_VAR(velocity_permutation.size());

  for (auto& tree_permutation : velocity_permutation) {
    PRINT_VEC(tree_permutation);
  }

  auto context = plant.CreateDefaultContext();
  MatrixXd M(nv, nv);
  plant.CalcMassMatrix(*context, &M);

  const BlockSparseMatrix<double> Mt =
      internal::ExtractBlockDiagonalMassMatrix(M, velocity_permutation);
  for (size_t t = 0; t < velocity_permutation.size(); ++t) {
    PRINT_VARn(Mt.get_block(t));
  }

  // All trees participate.
  std::vector<int> participating_trees(velocity_permutation.size());
  std::iota(participating_trees.begin(), participating_trees.end(), 0);
  VerifyMassMatrixReconstructionFromDiagonalBlocks(Mt, velocity_permutation,
                                                   participating_trees, M);

  // Verify body to tree map.
  const auto robot1 = plant.GetModelInstanceByName("robot1");
  const auto body_index = plant.GetBodyByName("iiwa_link_0", robot1).index();
  PRINT_VAR(body_to_tree_map[body_index]);
  PRINT_VEC(body_to_tree_map);
}

GTEST_TEST(DofsPermutation, AllegroHands) {
  const std::string right_hand_model_path = FindResourceOrThrow(
      "drake/manipulation/models/"
      "allegro_hand_description/sdf/allegro_hand_description_right.sdf");

  const std::string left_hand_model_path = FindResourceOrThrow(
      "drake/manipulation/models/"
      "allegro_hand_description/sdf/allegro_hand_description_left.sdf");

  MultibodyPlant<double> plant(0);
  Parser parser(&plant);
  const ModelInstanceIndex right_hand =
      parser.AddModelFromFile(right_hand_model_path, "right hand");
  const ModelInstanceIndex left_hand =
      parser.AddModelFromFile(left_hand_model_path, "left hand");

  // Weld the hands to the world frame
  (void)right_hand;
  // plant.WeldFrames(plant.world_frame(),
  //                 plant.GetFrameByName("hand_root", right_hand));
  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("hand_root", left_hand));
  plant.Finalize();

  const int nv = plant.num_velocities();
  auto context = plant.CreateDefaultContext();
  MatrixXd M(nv, nv);
  plant.CalcMassMatrix(*context, &M);
  // PRINT_VARn(M);

  // Compute BFT to DFT permutation.
  std::vector<std::vector<int>> velocity_permutation;
  std::vector<int> body_to_tree_map;
  const internal::MultibodyTreeTopology& topology =
      MultibodyPlantTester::get_topology(plant);
  drake::multibody::internal::ComputeBfsToDfsPermutation(
      topology, &velocity_permutation, &body_to_tree_map);

  PRINT_VAR(plant.num_velocities());
  PRINT_VAR(velocity_permutation.size());
  for (auto& tree_permutation : velocity_permutation) {
    PRINT_VEC(tree_permutation);
  }
  PRINT_VEC(body_to_tree_map);

  const BlockSparseMatrix<double> Mt =
      internal::ExtractBlockDiagonalMassMatrix(M, velocity_permutation);
  for (size_t t = 0; t < velocity_permutation.size(); ++t) {
    PRINT_VARn(Mt.get_block(t));
  }

  // All trees participate.
  std::vector<int> participating_trees(velocity_permutation.size());
  std::iota(participating_trees.begin(), participating_trees.end(), 0);
  VerifyMassMatrixReconstructionFromDiagonalBlocks(Mt, velocity_permutation,
                                                   participating_trees, M);

  // Verify body to tree map.
  // const auto robot1 = plant.GetModelInstanceByName("robot1");
  // const auto body_index = plant.GetBodyByName("iiwa_link_0", robot1).index();
  // PRINT_VAR(body_to_tree_map[body_index]);
  // PRINT_VEC(body_to_tree_map);

  // Right hand has a free base.
  BodyIndex body_index = plant.GetBodyByName("hand_root", right_hand).index();
  int t = body_to_tree_map[body_index];
  // Right hand tree is free. Therefore it has 16 + 6 = 22 dofs.
  EXPECT_EQ(velocity_permutation[t].size(), 22);
  for (int i = 0; i < 16; ++i) {
    const std::string link_name = fmt::format("link_{:d}", i);
    body_index = plant.GetBodyByName(link_name, right_hand).index();
    t = body_to_tree_map[body_index];
    ASSERT_GE(t, 0);  // We expect them not to be connected to the world.
    // Right hand tree is free. Therefore it has 16 + 6 = 22 dofs.
    EXPECT_EQ(velocity_permutation[t].size(), 22);
  }

  // Left hand has a base welded to the world. Therefore we expect each body to
  // be associated with the tree having 16 dofs.
  body_index = plant.GetBodyByName("hand_root", left_hand).index();
  t = body_to_tree_map[body_index];
  EXPECT_EQ(velocity_permutation[t].size(), 16);
  for (int i = 0; i < 16; ++i) {
    const std::string link_name = fmt::format("link_{:d}", i);
    body_index = plant.GetBodyByName(link_name, left_hand).index();
    t = body_to_tree_map[body_index];
    ASSERT_GE(t, 0);  // We expect them not to be connected to the world.
    // Right hand tree is free. Therefore it has 16 + 6 = 22 dofs.
    EXPECT_EQ(velocity_permutation[t].size(), 16);
  }
}

const RigidBody<double>& AddBox(std::string name, double size,
                                const Vector4<double>& color,
                                MultibodyPlant<double>* plant) {
  double density = 1000.0;
  const double point_radius = 0.005;
  const Vector4<double> cyan(0.0, 1.0, 1.0, 1.0);

  const double mass = density * size * size * size;
  const SpatialInertia<double> M_BBcm_B(
      mass, Vector3d::Zero(), UnitInertia<double>::SolidBox(size, size, size));

  const RigidBody<double>& body = plant->AddRigidBody(name, M_BBcm_B);

  geometry::Box box = geometry::Box::MakeCube(size);
  plant->RegisterVisualGeometry(body, RigidTransformd(), box, name + "_visual",
                                color);
  plant->RegisterCollisionGeometry(body, RigidTransformd(), box,
                                   name + "_collision",
                                   CoulombFriction<double>());

  // We "emulate multicontact" by adding small spheres.
  const int grid_size = 3;
  const geometry::Sphere sphere(point_radius);
  for (int i = 0; i < grid_size; ++i) {
    const double x = -size / 2.0 + i * size / (grid_size - 1);
    for (int j = 0; j < grid_size; ++j) {
      const double y = -size / 2.0 + j * size / (grid_size - 1);
      for (int k = 0; k < grid_size; ++k) {
        const double z = -size / 2.0 + k * size / (grid_size - 1);
        const std::string root_name =
            fmt::format("{}_{:d}{:d}{:d}_collision", name, i, j, k);
        const RigidTransformd X_BG(Vector3d(x, y, z));
        plant->RegisterVisualGeometry(body, X_BG, sphere, root_name + "_visual",
                                      cyan);
        plant->RegisterCollisionGeometry(body, X_BG, sphere,
                                         root_name + "_collision",
                                         CoulombFriction<double>());
      }
    }
  }

  return body;
}

void MakeStackWorld(MultibodyPlant<double>* plant) {
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> green(0.5, 1.0, 0.5, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  const Vector4<double> orange(1.0, 0.6, 0.0, 1.0);
  const Vector4<double> purple(0.9, 0.0, 0.88, 1.0);

  // Add a box fixed to the ground. Top is at z = 0.
  const auto& ground = AddBox("ground", 10.0, green, plant);
  plant->WeldFrames(plant->world_frame(), ground.body_frame(),
                    RigidTransformd(Vector3d(0.0, 0.0, -5.0)));

  // Add stack of boxes with "multicontact".
  AddBox("box1", 0.5, orange, plant);

  // We add a floating box here, not in contact, so that there is a
  // non-participating tree somewhere in the middle.
  AddBox("box4", 0.2, purple, plant);

  AddBox("box2", 0.25, blue, plant);
  AddBox("box3", 0.125, red, plant);

  // why not, add a kuka.
  const std::string iiwa_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_spheres_collision.urdf");
  Parser parser(&*plant);
  const ModelInstanceIndex robot_model =
      parser.AddModelFromFile(iiwa_path, "robot_arm");
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("base", robot_model),
                    RigidTransformd(Vector3d(-0.5, 0.0, 0.)));

  // iiwa_joint_1
}

void SetStackWorldState(const MultibodyPlant<double>& plant,
                        systems::Context<double>* context) {
  const double slop =
      1e-3;  // small slop used to avoid collision between boxes.

  const RigidTransformd X_WB1(Vector3d(0, 0, 0.25 + slop));
  plant.SetFreeBodyPoseInWorldFrame(context, plant.GetBodyByName("box1"),
                                    X_WB1);

  const RigidTransformd X_WB2 =
      X_WB1 * RigidTransformd(Vector3d(0, 0, 0.25 + 0.125 + slop));
  plant.SetFreeBodyPoseInWorldFrame(context, plant.GetBodyByName("box2"),
                                    X_WB2);

  const RigidTransformd X_WB3 =
      X_WB2 * RigidTransformd(Vector3d(0, 0, 0.125 + 0.0625 + slop));
  plant.SetFreeBodyPoseInWorldFrame(context, plant.GetBodyByName("box3"),
                                    X_WB3);

  const RigidTransformd X_WB4 = X_WB2 * RigidTransformd(Vector3d(0.75, 0, 0.0));
  plant.SetFreeBodyPoseInWorldFrame(context, plant.GetBodyByName("box4"),
                                    X_WB4);

  // Make the tip of the robot touch the box.
  const auto& joint4 = plant.GetJointByName<RevoluteJoint>("iiwa_joint_4");
  joint4.set_angle(context, -M_PI / 4 * 1.5);

  // N.B. In this configuration the iiwa_link_5 and iiwa_link_7 are in contact.
  // Therefore we have one extra "unexpected" contact.
  const auto& joint6 = plant.GetJointByName<RevoluteJoint>("iiwa_joint_6");
  joint6.set_angle(context, M_PI / 4);
}

GTEST_TEST(ContactPermutation, Stack) {
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.0);

  MakeStackWorld(&plant);

  plant.Finalize();

  // Connect visualizer. Useful for when this test is used for debugging.
  geometry::DrakeVisualizerParams viz_params;
  viz_params.role = geometry::Role::kIllustration;  // kProximity;
  drake::geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph,
                                                  nullptr, viz_params);

  auto diagram = builder.Build();

  auto context = diagram->CreateDefaultContext();
  auto& plant_context = plant.GetMyMutableContextFromRoot(context.get());

  // Set state.
  SetStackWorldState(plant, &plant_context);

  // Publish for visualization.
  diagram->Publish(*context);

  const std::vector<geometry::PenetrationAsPointPair<double>>& point_pairs =
      plant.EvalPointPairPenetrations(plant_context);

  PRINT_VAR(point_pairs.size());

  // We need to know what tree each body belongs to and thus we compute BFT to
  // DFT permutation.
  std::vector<std::vector<int>> velocity_permutation;
  std::vector<int> body_to_tree_map;
  const internal::MultibodyTreeTopology& topology =
      MultibodyPlantTester::get_topology(plant);
  drake::multibody::internal::ComputeBfsToDfsPermutation(
      topology, &velocity_permutation, &body_to_tree_map);

  std::vector<SortedPair<int>> contacts;
  const auto& inspector = scene_graph.model_inspector();
  for (const auto& pp : point_pairs) {
    const geometry::FrameId frameA = inspector.GetFrameId(pp.id_A);
    const geometry::FrameId frameB = inspector.GetFrameId(pp.id_B);
    const BodyIndex bodyA = plant.GetBodyFromFrameId(frameA)->index();
    const BodyIndex bodyB = plant.GetBodyFromFrameId(frameB)->index();
    const int treeA = body_to_tree_map[bodyA];
    const int treeB = body_to_tree_map[bodyB];
    // SceneGraph does not report collisions between anchored geometries.
    // We verify this.
    ASSERT_FALSE(treeA < 0 && treeB < 0);
    contacts.push_back({treeA, treeB});
  }

  const int num_trees = velocity_permutation.size();
  std::vector<int> participating_trees;
  const internal::ContactGraph graph =
      internal::ComputeContactGraph(num_trees, contacts, &participating_trees);
  PRINT_VAR(num_trees);
  PRINT_VAR(Eigen::Map<VectorX<int>>(participating_trees.data(),
                                     participating_trees.size()));

  // We expect the box-box patches to have 11 contact (when grid_size=3) since
  // we have:
  //  1. 9 spheres of the upper box vs. face of the bottom box.
  //  2. sphere of bottom box with face of upper box.
  //  3. sphere of bottom box with sphere of upper box.
  PRINT_VAR(graph.patches.size());
  for (const auto& p : graph.patches) {
    PRINT_VAR(p.t1);
    PRINT_VAR(p.t2);
    PRINT_VEC(p.contacts);
    for (int k : p.contacts) {
      const auto& pp = point_pairs[k];
      const geometry::FrameId frameA = inspector.GetFrameId(pp.id_A);
      const geometry::FrameId frameB = inspector.GetFrameId(pp.id_B);
      const auto& bodyA = *plant.GetBodyFromFrameId(frameA);
      const auto& bodyB = *plant.GetBodyFromFrameId(frameB);
      PRINT_VAR("(" + bodyA.name() + ", " + bodyB.name() + ")");
    }
  }

  const MatrixXd Jc =
      MultibodyPlantTester::EvalContactJacobian(plant, plant_context);

  const BlockSparseMatrix<double> J_blocks = ExtractBlockJacobian(
      Jc, graph, velocity_permutation, participating_trees);

  for (const auto& [p, t, Jpt] : J_blocks.get_blocks()) {
    PRINT_VAR(p);
    PRINT_VAR(t);
    PRINT_VARn(Jpt);
  }

  VerifyJacobianReconstructionFromBlocks(J_blocks, graph, velocity_permutation,
                                         participating_trees, Jc);

  const int nv = plant.num_velocities();
  MatrixXd M(nv, nv);
  plant.CalcMassMatrix(plant_context, &M);

  const BlockSparseMatrix<double> Mt = internal::ExtractBlockDiagonalMassMatrix(
      M, velocity_permutation, participating_trees);
  PRINT_VAR(Mt.num_blocks());
  PRINT_VAR(participating_trees.size());
  for (int t = 0; t < Mt.num_blocks(); ++t) {
    PRINT_VARn(Mt.get_block(t));
  }

  VerifyMassMatrixReconstructionFromDiagonalBlocks(Mt, velocity_permutation,
                                                   participating_trees, M);
}

}  // namespace
}  // namespace multibody
}  // namespace drake
