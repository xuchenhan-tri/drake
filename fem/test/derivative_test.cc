#include <cmath>
#include <functional>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include "drake/fem/fem_config.h"
#include "drake/fem/fem_data.h"
#include "drake/fem/fem_force.h"
#include "drake/fem/mesh_utility.h"

namespace drake {
namespace fem {
namespace {

template <typename InputType, typename OutputType>
class DerivativeTester {
 public:
  static double step_size() {
    // Step size is set to ε^(1/3) according to the advise in
    // https://www.cs.ucr.edu/~craigs/papers/2019-derivatives/course.pdf.
    return std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);
  }

  /** The function `compute' should provide the differential of the function,
   * i.e. df = f'(x) * dx. */
  void RunTest(
      const std::function<void(const InputType& x, const InputType& dx,
                               OutputType* f, OutputType* df)>& compute) {
    const double h = step_size();
    // Evaluate the function value the the function differential at x and
    // x+h*dx.
    compute(x_, dx_, &f0_, &df0_);
    compute(x_ + h * dx_, dx_, &f1_, &df1_);

    // Taylor expand f at x gives:
    //
    //     f(x+h*dx) = f(x) + h * f'(x) * dx +  1/2 * h² * f''(x) * dx² +  1/6 *
    //     h³ * f'''(x) * dx³ + O(h⁴)
    //
    // for some ξ.
    //
    // Taylor expand f' at x gives:
    //
    //     f'(x+h*dx) = f'(x) + h * f''(x) * dx +  1/2 * h² * f'''(η) * dx² +
    //     O(h³)
    //
    // for some η.
    // So when computed analytically,
    //
    //     (f(x+h*dx) - f(x))/h - (f'(x+h*dx)*dx + f'(x)*dx)/2 =
    //     -1/12  * h² * f'''(x) * dx³ +  O(h³)
    //
    // In actual numerical calculation (f(x+h*dx) - f(x))/h gives round-off
    // error of around machine_epsilon*|f(x)| in the subtraction, which is
    // scaled by the division of h. Plugging in h = machine_epsilon ^ (1/3), we
    // get the round off error is ~ |f(x)| * h².
    //
    // Combining the two sources of error, we get
    //
    //     (f(x+h*dx) - f(x))/h - (f'(x+h*dx)*dx + f'(x)*dx)/2 <= (|f(x)| + 1/12
    //     * |f'''(x)|) * h² +  O(h³)
    //
    // Therefore we should expect a residual of order h².
    // When the derivative is correct, the leading order will be O(1). Hence we
    // set the tolerance to be on order of h.

    OutputType difference = (f1_ - f0_) / h;
    OutputType differential = (df1_ + df0_) / 2.0;
    // Setting the scale to be ~2*|f(x)|, which is used as a proxy of (|f(x)| +
    // 1/12 * |f'''(x)|). In the unlikely event where |f(x)|/|f'''(x)| is on the
    // order of h, the test may produce a false failure.
    double scale = std::max(f0_.norm() + f1_.norm(), 1.0);
    double residual = (difference - differential).norm();
    EXPECT_NEAR(residual, 0.0, scale * h);
  }

  void set_x(const InputType& x) { x_ = x; }
  void set_dx(const InputType& dx) { dx_ = dx; }
  const OutputType& get_f0() const { return f0_; }
  const OutputType& get_df0() const { return df0_; }
  const OutputType& get_f1() const { return f1_; }
  const OutputType& get_df1() const { return df1_; }

 private:
  InputType x_;
  InputType dx_;
  OutputType f0_;
  OutputType df0_;
  OutputType f1_;
  OutputType df1_;
};

class FemForceTest : public ::testing::Test {
 public:
  void SetUp() override {
    // dt does not matter here, setting it to an arbitrary value of 0.1.
    data_ = std::make_unique<FemData<double>>(0.1);
    dut_ = std::make_unique<FemForce<double>>(data_->get_elements());
    MaterialConfig config;
    config.density = 1e3;
    config.youngs_modulus = 1e4;
    config.poisson_ratio = 0.4;
    config.mass_damping = 0.3;
    config.stiffness_damping = 0.5;
    const int nx = 1;
    const int ny = 1;
    const int nz = 1;
    const double h = 0.1;
    const auto positions =
        MeshUtility<double>::AddRectangularBlockVertices(nx, ny, nz, h);
    const auto indices =
        MeshUtility<double>::AddRectangularBlockMesh(nx, ny, nz);
    data_->AddUndeformedObject(indices, positions, config);
  }

 protected:
  std::unique_ptr<FemForce<double>> dut_;  // The device under test.
  std::unique_ptr<FemData<double>> data_;
};

TEST_F(FemForceTest, EnergyDerivative) {
  // Move vertices to random positions.
  Matrix3X<double> positions =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  Matrix3X<double> tmp_dx =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  tmp_dx.normalize();
  DerivativeTester<Matrix3X<double>, Vector1<double>> energy_derivative_tester;
  energy_derivative_tester.set_x(positions);
  energy_derivative_tester.set_dx(tmp_dx);
  // Set q_n to be `positions' and update the states depending on q_n. This
  // mimics the condition under which we usually do energy and force
  // calculations, but it is unnecessary for the test to pass.
  auto& elements = data_->get_mutable_elements();
  for (auto& e : elements) {
    e.UpdateTimeNPositionBasedState(positions);
  }
  auto compute = [this](const Matrix3X<double>& x, const Matrix3X<double>& dx,
                        Vector1<double>* f, Vector1<double>* df) {
    for (auto& e : data_->get_mutable_elements()) {
      e.UpdateF(x);
    }
    (*f)(0) = dut_->CalcElasticEnergy();
    Matrix3X<double> elastic_force(3, x.cols());
    elastic_force.setZero();
    // Force is the negative gradient of energy, so we scale by -1 to get the
    // derivative of the energy.
    dut_->AccumulateScaledElasticForce(-1, &elastic_force);
    (*df)(0) = (elastic_force.array() * dx.array()).sum();
  };
  energy_derivative_tester.RunTest(compute);
}

TEST_F(FemForceTest, ForceDifferential) {
  // Move vertices to random positions.
  Matrix3X<double> positions =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  Matrix3X<double> tmp_dx =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  tmp_dx.normalize();
  DerivativeTester<Matrix3X<double>, Matrix3X<double>>
      force_differential_tester;
  force_differential_tester.set_x(positions);
  force_differential_tester.set_dx(tmp_dx);
  // Set q_n to be `positions' and update the states depending on q_n. This
  // mimics the condition under which we usually do force and force differential
  // calculations, but it is unnecessary for the test to pass.
  for (auto& e : data_->get_mutable_elements()) {
    e.UpdateTimeNPositionBasedState(positions);
  }
  auto compute = [this](const Matrix3X<double>& x, const Matrix3X<double>& dx,
                        Matrix3X<double>* f, Matrix3X<double>* df) {
    auto& elements = data_->get_mutable_elements();
    for (auto& e : elements) {
      e.UpdateF(x);
    }
    f->resize(3, x.cols());
    f->setZero();
    dut_->AccumulateScaledElasticForce(1, f);
    df->resize(3, x.cols());
    df->setZero();
    dut_->AccumulateScaledElasticForceDifferential(1, dx, df);
  };
  force_differential_tester.RunTest(compute);
}

TEST_F(FemForceTest, ForceDerivative) {
  // Move vertices to random positions.
  Matrix3X<double> positions =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  Matrix3X<double> tmp_dx =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  tmp_dx.normalize();
  DerivativeTester<Matrix3X<double>, Matrix3X<double>> force_derivative_tester;
  force_derivative_tester.set_x(positions);
  force_derivative_tester.set_dx(tmp_dx);
  // Set q_n to be `positions' and update the states depending on q_n. This
  // mimics the condition under which we usually do force and force differential
  // calculations, but it is unnecessary for the test to pass.
  for (auto& e : data_->get_mutable_elements()) {
    e.UpdateTimeNPositionBasedState(positions);
  }
  auto compute = [this](const Matrix3X<double>& x, const Matrix3X<double>& dx,
                        Matrix3X<double>* f, Matrix3X<double>* df) {
    auto& elements = data_->get_mutable_elements();
    for (auto& e : elements) {
      e.UpdateF(x);
    }
    f->resize(3, x.cols());
    f->setZero();
    dut_->AccumulateScaledElasticForce(1, f);
    Eigen::SparseMatrix<double> K(x.size(), x.size());
    std::vector<Eigen::Triplet<double>> non_zero_entries;
    non_zero_entries.clear();
    dut_->SetSparsityPattern(&non_zero_entries);
    K.setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());
    K.makeCompressed();
    // Stiffness matrix is the second derivative of the energy density and thus
    // the negative derivative of the force, so we scale by -1 here.
    dut_->AccumulateScaledStiffnessMatrix(-1, &K);
    const VectorX<double>& dx_tmp =
        Eigen::Map<const VectorX<double>>(dx.data(), dx.size());
    VectorX<double> df_tmp = K * dx_tmp;
    *df = Eigen::Map<Matrix3X<double>>(df_tmp.data(), 3, df_tmp.size() / 3);
  };
  force_derivative_tester.RunTest(compute);
}

TEST_F(FemForceTest, Damping) {
  // Move vertices to random positions.
  Matrix3X<double> positions =
      Matrix3X<double>::Random(3, data_->get_num_vertices());
  Matrix3X<double> dv = Matrix3X<double>::Random(3, data_->get_num_vertices());
  for (auto& e : data_->get_mutable_elements()) {
    e.UpdateTimeNPositionBasedState(positions);
    e.UpdateF(positions);
  }
  Matrix3X<double> df_matrix_free(3, data_->get_num_vertices());
  Matrix3X<double> df_direct(3, data_->get_num_vertices());
  // Calculate the damping force differential through the matrix free method.
  df_matrix_free.setZero();
  dut_->AccumulateScaledDampingForceDifferential(1, dv, &df_matrix_free);

  // Calculate the damping force differential through direct multiplication.
  Eigen::SparseMatrix<double> D(positions.size(), positions.size());
  std::vector<Eigen::Triplet<double>> non_zero_entries;
  non_zero_entries.clear();
  dut_->SetSparsityPattern(&non_zero_entries);
  D.setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());
  D.makeCompressed();
  // Damping force differential = -D * dv.
  dut_->AccumulateScaledDampingMatrix(-1, &D);
  const VectorX<double>& dv_tmp =
      Eigen::Map<const VectorX<double>>(dv.data(), dv.size());
  VectorX<double> df_direct_tmp = D * dv_tmp;
  df_direct = Eigen::Map<Matrix3X<double>>(df_direct_tmp.data(), 3,
                                           df_direct_tmp.size() / 3);
  double scale = std::max(1.0, df_direct.norm() + df_matrix_free.norm());
  EXPECT_NEAR((df_direct - df_matrix_free).norm(), 0.0,
              scale * std::numeric_limits<double>::epsilon());
}
}  // namespace
}  // namespace fem
}  // namespace drake
