#include "drake/multibody/fixed_fem/dev/contact_data_calculator.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/unused.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/fixed_fem/dev/collision_object.h"
#include "drake/multibody/math/spatial_algebra.h"

namespace drake {
namespace multibody {
namespace fixed_fem {
namespace internal {
namespace {

using contact_solvers::internal::ExtractNormal;
using contact_solvers::internal::ExtractTangent;
using contact_solvers::internal::LinearOperator;
using contact_solvers::internal::PointContactData;
using geometry::Box;
using geometry::SurfaceMesh;
using geometry::VolumeMesh;
using math::RigidTransform;

/* The objects in contact look like this:
                +Z
      B          |          C
        -------  |   -------
        |     |  |   |     |
        |  ---+--+---+---  |
        |  |  ●  |   ●  |  |
        |  |  |  |A  |  |  |
  -Y----+--+--+--+---+--+--+----+Y
        |  |  |  |   |  |  |
        |  |  ●  |   ●  |  |
        |  ---+--+---+---  |
        |     |  |   |     |
        -------  |   -------
                 |
                -Z
  where object A is deformable and object B and C are rigid. A is a cube with
  side length 2 centered at origin. B and C are boxes with size (4,1,4),
  centered at (0, -1, 0) and (0, 1, 0) respectively. */
template <typename T>
const VolumeMesh<T> MakeUnitBoxTetMesh() {
  const double length = 2;
  const Box box(length, length, length);
  const VolumeMesh<T> mesh =
      geometry::internal::MakeBoxVolumeMesh<T>(box, length);
  return mesh;
}

template <typename T>
CollisionObject<T> MakeCollisionObject(const RigidTransform<T>& X_WG,
                                       const SpatialVelocity<T>& V_WG) {
  const double Lx = 4;
  const double Ly = 1;
  const double Lz = 4;
  const double dx = 1;
  const Box box(Lx, Ly, Lz);
  const VolumeMesh<double> volume_mesh =
      geometry::internal::MakeBoxVolumeMesh<double>(box, dx);
  const SurfaceMesh<double> surface_mesh =
      geometry::internal::ConvertVolumeToSurfaceMesh(volume_mesh);
  geometry::ProximityProperties prox_prop;
  double dummy_elastic_modulus(1);
  double dummy_dissipation(0);
  double dummy_point_stiffness(1);
  geometry::AddContactMaterial(dummy_elastic_modulus, dummy_dissipation,
                               dummy_point_stiffness, CoulombFriction<double>(),
                               &prox_prop);
  const auto motion_callback = [=](const T& time, math::RigidTransform<T>* pose,
                                   SpatialVelocity<T>* spatial_velocity) {
    unused(time);
    *pose = X_WG;
    *spatial_velocity = V_WG;
  };
  return CollisionObject<T>(surface_mesh, prox_prop, motion_callback);
}

/* Limited test to show correctness of the result from ContactDataCalculator.
 Given a simple, tractable set of input objects in contact, we analytically
 compute the expected contact velocities and compare it against the result from
 ContactDataCalculator. */
template <typename T>
void TestContactDataCalculator() {
  // constexpr double kEps = std::numeric_limits<double>::epsilon();
  const Vector3<T> v_WA(0, 0, 1);
  const Vector3<T> w_WB(0, 0, 0);
  const Vector3<T> v_WB(0, 1, 0);
  const Vector3<T> p_WB(0, -1, 0);
  const Vector3<T> w_WC(1, 0, 0);
  const Vector3<T> v_WC(0, 0, 0);
  const Vector3<T> p_WC(0, 1, 0);
  const VolumeMesh<T> deformable_mesh = MakeUnitBoxTetMesh<T>();
  CollisionObject<T> rigid_B = MakeCollisionObject(
      RigidTransform<T>(p_WB), SpatialVelocity<T>(w_WB, v_WB));
  CollisionObject<T> rigid_C = MakeCollisionObject(
      RigidTransform<T>(p_WC), SpatialVelocity<T>(w_WC, v_WC));
  /* Ensure that the pose and the spatial velocity of the rigid object B and C
   are set to their prescribed values. */
  const T dummy_time(0);
  rigid_B.UpdatePositionAndVelocity(dummy_time);
  rigid_C.UpdatePositionAndVelocity(dummy_time);

  /* Set up the generalized velocities. */
  VectorX<T> generalized_velocities(deformable_mesh.num_vertices() * 3 + 12);
  for (int i = 0; i < deformable_mesh.num_vertices(); ++i) {
    /* The deformable object A has velocity v_WA = (0, 0, 1). */
    generalized_velocities.template segment<3>(3 * i) = v_WA;
  }
  /* Attach the velocities of rigid object B and C. */
  generalized_velocities.template segment<3>(deformable_mesh.num_vertices() *
                                             3) = w_WB;
  generalized_velocities.template segment<3>(
      deformable_mesh.num_vertices() * 3 + 3) = v_WB;
  generalized_velocities.template segment<3>(
      deformable_mesh.num_vertices() * 3 + 6) = w_WC;
  generalized_velocities.template segment<3>(
      deformable_mesh.num_vertices() * 3 + 9) = v_WC;

  const DeformableContactSurface<T> contact_surface_AB =
      ComputeTetMeshTriMeshContact(deformable_mesh, rigid_B.mesh(),
                                   rigid_B.pose());
  const DeformableContactSurface<T> contact_surface_AC =
      ComputeTetMeshTriMeshContact(deformable_mesh, rigid_C.mesh(),
                                   rigid_C.pose());
  /* A sanity check: The number of contacts should be the same for both contact
   surfaces due to symmetry. */
  const int nc_AB = contact_surface_AB.num_polygons();
  const int nc_AC = contact_surface_AB.num_polygons();
  EXPECT_EQ(nc_AB, nc_AC);

  ContactDataCalculator<T> contact_data_calculator;
  const PointContactData<T> point_contact_data =
      contact_data_calculator.ComputeContactData(deformable_mesh,
                                                 {rigid_B, rigid_C});

  const LinearOperator<T>& Jc = point_contact_data.get_Jc();
  const int nc = nc_AB + nc_AC;
  VectorX<T> contact_velocities(3 * nc);
  Jc.Multiply(generalized_velocities, &contact_velocities);

  VectorX<T> vn(nc);
  VectorX<T> vt(2 * nc);
  ExtractNormal(contact_velocities, &vn);
  ExtractTangent(contact_velocities, &vt);
  /* For contact between A and B, the normal component is expected to be -1 (due
   to the motion of B) and the tangnent component is expected to have norm 1
   (due to the motion of A). */
  for (int i = 0; i < nc_AB; ++i) {
    if constexpr (std::is_same_v<T, double>) {
      EXPECT_DOUBLE_EQ(vn(i), -1);
      EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * i).norm(), v_WA.norm());
    } else {
      EXPECT_DOUBLE_EQ(vn(i).value(), -1);
      EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * i).norm().value(),
                       v_WA.norm().value());
    }
  }
  for (int i = 0; i < nc_AC; ++i) {
    const int contact_index = nc_AB + i;
    const ContactPolygonData<T>& data = contact_surface_AC.polygon_data(i);
    /* This is by definition the position of the contact point Cq measured in
     and expressed in the deformable frame, but we are using the assumption
     that the deformable frame is always the same as the world frame for now,
     and thus we denote the quantity with p_WCq. This assumption might change in
     the future. */
    const Vector3<T>& p_WCq = data.centroid;
    const Vector3<T> p_CoCq_W = p_WCq - p_WC;
    const Vector3<T> v_WCq = rigid_C.translational_velocity() +
                             rigid_C.rotational_velocity().cross(p_CoCq_W);
    const T vn_WCq = v_WCq.dot(data.unit_normal);
    const Vector3<T> vt_WCq = v_WA - (v_WCq - data.unit_normal * vn_WCq);
    if constexpr (std::is_same_v<T, double>) {
      EXPECT_DOUBLE_EQ(vn(contact_index), -vn_WCq);
      EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * contact_index).norm(),
                       vt_WCq.norm());
    } else {
      EXPECT_DOUBLE_EQ(vn(contact_index).value(), -vn_WCq.value());
      EXPECT_DOUBLE_EQ(vt.template segment<2>(2 * contact_index).norm().value(),
                       vt_WCq.norm().value());
    }
  }
}

GTEST_TEST(ContactDataCalculatorTest, Double) {
  TestContactDataCalculator<double>();
}

GTEST_TEST(ContactDataCalculatorTest, AutoDiff) {
  TestContactDataCalculator<AutoDiffXd>();
}
}  // namespace
}  // namespace internal
}  // namespace fixed_fem
}  // namespace multibody
}  // namespace drake
