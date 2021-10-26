#include "drake/multibody/contact_solvers/geodesic_interior_point_method.h"
using conex::BlockMatrixTriplet;
using Eigen::VectorXd;
using std::vector;

#define USE_SCALED_JORDAN_MULT 1
// When defined, we use the Jordan multiplication
//  x * y = 1/sqrt(2) * ( x'y, x0 y1 + y0 x1 ).
//
//  The idempotents have form
//
//  1/sqrt(2) (1, y) *  1/sqrt(2) (1, y).
//
//  So spectral functions have form
//
//   1/sqrt(2) * (  exp(lambda_1/sqrt 2)  + exp(lambda_2/sqrt 2) )


namespace {

const double line_search_tolerance = 0.02;

vector<int> CumulativeSum(const vector<int>& x, int N) {
  vector<int> y(N + 1);
  y.at(0) = 0;
  for (int i = 1; i < N + 1; i++) {
    y.at(i) = y.at(i - 1) + x.at(i - 1);
  }
  return y;
}

inline Eigen::VectorXd Vect(const SpinFactorProduct& x) {
  Eigen::VectorXd y(x.w0.size() * 3);
  int offset = 0;
  for (int i = 0; i < x.w0.size(); i++) {
    y(offset + 2) = x.w0(i);
    y.segment(offset, 2) = x.w1.col(i);
    offset += 3;
  }
  return y;
}


inline double square_root(const double& x) { return std::sqrt(std::fabs(x)); }


template <typename T>
void Exp(double norm_x1, double* x0, T* x1) {
  double k = norm_x1;
#if USE_SCALED_JORDAN_MULT
  double lam1 = 1.0/std::sqrt(2.0) * (*x0 + k);
  double lam2 = 1.0/std::sqrt(2.0) * (*x0 - k);
  double scale = 1.0/std::sqrt(2.0);
#else
  double lam1 = *x0 + k;
  double lam2 = *x0 - k;
  double scale = .5;
#endif

  if (k > 0) {
    (*x1) *= scale * (exp(lam1) - exp(lam2)) / k;
  }
  (*x0) = (scale * (exp(lam1) + exp(lam2)));
}

template <typename T>
void Sqrt(double norm_x1, double* x0, T* x1) {
  double k = norm_x1;
#if USE_SCALED_JORDAN_MULT
  double lam1 = 1.0/std::sqrt(2.0) * (*x0 + k);
  double lam2 = 1.0/std::sqrt(2.0) * (*x0 - k);
  double scale = 1.0/std::sqrt(2.0);
#else
  double lam1 = *x0 + k;
  double lam2 = *x0 - k;
  double scale = 0.5;
#endif

  if (k > 0) {
    (*x1) *= scale * (square_root(lam1) - square_root(lam2)) / k;
  }
  (*x0) = (scale * (square_root(lam1) + square_root(lam2)));

#ifndef NDEBUG
  if (*x0 - x1->norm() < 0) {
    throw std::runtime_error("Square root not inside cone");
  }
#endif
}



template <typename T1, typename T2, typename T3>
void QuadraticRepresentation(double x1_norm_squared,
                             double inner_product_of_x1_and_y1, double x0,
                             const T1& x1, double y0, const T2& y1,
                             double* z_q0, T3* z_q1) {
  // We use the formula from Example 11.12 of "Formally Real Jordan Algebras
  // and Their Applications to Optimization"  by Alizadeh, which states the
  // quadratic representation of x equals the linear map
  //                          2xx' - (det x) * R
  // where R is the reflection operator R = diag(1, -1, ..., -1) and det x is
  // the determinate of x = (x0, x1), i.e., det x = x0^2 - |x1|^2.
  double det_x = x0 * x0 - x1_norm_squared;
  double scale = 2 * (x0 * y0 + inner_product_of_x1_and_y1);
  *z_q0 = scale * x0 - det_x * y0;
  z_q1->noalias() = scale * x1 + det_x * y1;

#if USE_SCALED_JORDAN_MULT
  *z_q0 *= 0.5;
  *z_q1 *= 0.5;
#endif

}

}  // namespace

double NormInf(const SpinFactorProduct& x) {
  VectorXd norms = x.w1.array().square().colwise().sum().sqrt();
  VectorXd lambda1 = x.w0 - norms;
  VectorXd lambda2 = x.w0 + norms;

  double y1 = lambda1.array().abs().maxCoeff();
  double y2 = lambda2.array().abs().maxCoeff();
#if USE_SCALED_JORDAN_MULT
  y1 *= 1.0/std::sqrt(2.0);
  y2 *= 1.0/std::sqrt(2.0);
#endif

  if (y1 > y2) {
    return y1;
  } else {
    return y2;
  }
}

bool TakeStep(const SpinFactorProduct& wa, const SpinFactorProduct& da,
              SpinFactorProduct* ya, bool affine_step) {

  SpinFactorElement expd;
  SpinFactorElement d;
  SpinFactorElement wsqrt;
  for (int i = 0; i < wa.w0.size(); i++) {
    SpinFactorElement w(wa, i);
    SpinFactorElement dt(da, i); d = dt;
    SpinFactorElement y;

    wsqrt = w;
    Sqrt(w.w1.norm(), &wsqrt.w0, &wsqrt.w1);

    double wsqrt_q1_norm_sqr = wsqrt.w1.squaredNorm();


    if (affine_step) {
      // Affine step: Q(w^1/2)(e + d)
      SpinFactorElement d_plus_e;
      d_plus_e.w0 = d.w0;
      d_plus_e.w1 = d.w1;
#if USE_SCALED_JORDAN_MULT
      d_plus_e.w0 += std::sqrt(2.0);
#else
      d_plus_e.w0 += 1;
#endif
      QuadraticRepresentation(wsqrt_q1_norm_sqr, wsqrt.w1.dot(d_plus_e.w1), wsqrt.w0,
                            wsqrt.w1, d_plus_e.w0, d_plus_e.w1, &y.w0, &y.w1);
#ifndef NDEBUG
      //if (d_plus_e.w0 - d_plus_e.w1.norm() < -line_search_tolerance) {
      if (d_plus_e.w0 - d_plus_e.w1.norm() < -0.03) {
        throw std::runtime_error("Affine update not inside cone");
      }
#endif
   } else {

      expd.w0 = d.w0;
      expd.w1 = d.w1;
      Exp(d.w1.norm(), &expd.w0, &expd.w1);
#ifndef NDEBUG
      if (expd.w0 - expd.w1.norm() < 0) {
        throw std::runtime_error("exp d not inside cone");
      }
#endif

      QuadraticRepresentation(wsqrt_q1_norm_sqr, wsqrt.w1.dot(expd.w1), wsqrt.w0,
                              wsqrt.w1, expd.w0, expd.w1, &y.w0, &y.w1);

#ifndef NDEBUG
      if (y.w0 - y.w1.norm() < 0) {
        DUMP(y.w0 - y.w1.norm());
        throw std::runtime_error("Geodesic update not inside cone");
      }
#endif
   }

    ya->w0(i) = y.w0;
    ya->w1.col(i) = y.w1;

  }
  return true;
}

SparseJacobian::SparseJacobian(const std::vector<BlockMatrixTriplet>& blocks)
    : blocks_(blocks),
      row_offset_(blocks_.size()),
      col_offset_(blocks_.size()),
      row_block_sizes_(blocks_.size()),
      col_block_sizes_(blocks_.size()) {
  for (const auto& b : blocks_) {
    int i = std::get<0>(b);
    int j = std::get<1>(b);
    row_block_sizes_.at(i) = std::get<2>(b).rows();
    col_block_sizes_.at(j) = std::get<2>(b).cols();
    if (j >= num_col_blocks_) {
      num_col_blocks_ = j + 1;
    }
    if (i >= num_row_blocks_) {
      num_row_blocks_ = i + 1;
    }
  }
  col_offset_ = CumulativeSum(col_block_sizes_, num_col_blocks_);
  row_offset_ = CumulativeSum(row_block_sizes_, num_row_blocks_);
}

void SparseJacobian::RescaleByFrictionCoefficient(const VectorXd& mu) {
  for (auto& Ji : blocks_) {
    int row_block = std::get<0>(Ji);
    int contact_number = row_offset_.at(row_block) / 3;
    int num_contacts = row_block_sizes_.at(row_block) / 3;
    for (int j = 0; j < num_contacts; j++) {
      std::get<2>(Ji).row(2  + j * 3) *= 1.0/mu(contact_number + j);
    }
  }
}

VectorXd SparseJacobian::Multiply(const VectorXd& x) const {
  VectorXd y(row_offset_.at(num_row_blocks_));
  y.setZero();
  for (const auto& b : blocks_) {
    int i = std::get<0>(b);
    int j = std::get<1>(b);
    y.segment(row_offset_.at(i), row_block_sizes_.at(i)) +=
        std::get<2>(b) * x.segment(col_offset_.at(j), col_block_sizes_.at(j));
  }
  return y;
}

VectorXd SparseJacobian::MultiplyByTranspose(const VectorXd& x) const {
  VectorXd y(col_offset_.at(num_col_blocks_));
  y.setZero();
  for (const auto& b : blocks_) {
    int i = std::get<0>(b);
    int j = std::get<1>(b);
    y.segment(col_offset_.at(j), col_block_sizes_.at(j)) +=
        std::get<2>(b).transpose() *
        x.segment(row_offset_.at(i), row_block_sizes_.at(i));
  }
  return y;
}

VectorXd BlockDiagonalMatrix::Multiply(const VectorXd& x) const {
  int r = 0;
  VectorXd b(x.rows());
  for (size_t i = 0; i < blocks_.size(); i++) {
    int s = blocks_.at(i).rows();
    b.segment(r, s) = blocks_.at(i) * x.segment(r, s);
    r += s;
  }
  return b;
}

void NewtonDirection::FactorNewtonSystem() {
  kkt_solver_.SetWeightMatrix(QWQ());
  kkt_solver_.Factor();
  newton_system_factored_ = true;
}

void NewtonDirection::SolveNewtonSystem(double sqrt_inv_mu,
                                  const VectorXd& vstar,
                                  const VectorXd& v_hat,
                                  VectorXd* v) {
  int r = 0;
  Vector3d e_i = Vector3d::Zero();
#if USE_SCALED_JORDAN_MULT
  e_i(2) = std::sqrt(2.0);
#else
  e_i(2) = 1;
#endif
  VectorXd t1(num_constraints_ * 3);
  VectorXd t2(num_constraints_ * 3);

  VectorXd w_vect(3);
  for (int i = 0; i < num_constraints_; i++) {
    w_vect(2) = w_.w0(i);
    w_vect.segment(0, 2) = w_.w1.col(i);

#ifndef NDEBUG
    if (w_vect(2) < w_vect.segment(0, 2).norm()) {
      throw std::runtime_error("W not inside cone.");
    }
#endif

    t1.segment(r, 3) = w_vect + Qwsqrt().at(i) * W_.at(i) * (e_i);
    t2.segment(r, 3) =
        -sqrt_inv_mu * v_hat.segment(r, 3) + R_.at(i).asDiagonal() * w_vect;
    t2.segment(r, 3) = QWQ().at(i) * t2.segment(r, 3);
    r += 3;
  }

  *v = M_.Multiply(sqrt_inv_mu * vstar);

  *v += J_.MultiplyByTranspose(t1 - t2);

  if (newton_system_factored_ == false) {
    FactorNewtonSystem();
  }
  kkt_solver_.SolveInPlace(v);
}

void NewtonDirection::CalculateNewtonDirection(double sqrt_inv_mu,
                                               const VectorXd& vstar,
                                               const VectorXd& v_hat,
                                               VectorXd* d, VectorXd* v) {
  SolveNewtonSystem(sqrt_inv_mu, vstar, v_hat, v);

  auto Q = Qwsqrt();
  VectorXd Jv_times_sqrt_inv_mu = J_.Multiply(*v);
  Vector3d e_i = Vector3d::Zero();
#if USE_SCALED_JORDAN_MULT
  e_i(2) = std::sqrt(2.0);
#else
  e_i(2) = 1;
#endif

  d->resize(3 * num_constraints_);

  int r = 0;
  VectorXd wvect_i(3);
  for (int i = 0; i < num_constraints_; i++) {
    const auto& vhat_i = v_hat.segment(r, 3);
    wvect_i(2) = w_.w0(i);
    wvect_i.segment(0, 2) = w_.w1.col(i);

    const auto& Jv_i = Jv_times_sqrt_inv_mu.segment(r, 3);
    Matrix3d W_i =
        (Matrix3d::Identity() + Q.at(i) * R_.at(i).asDiagonal() * Q.at(i))
            .inverse();
    d->segment(r, 3) =
        W_i * (e_i - Q.at(i) * (Jv_i - sqrt_inv_mu * vhat_i +
                                R_.at(i).asDiagonal() * wvect_i));
    r += 3;
  }
}


void NewtonDirection::SetW(const SpinFactorProduct& w) {
  w_ = w;

#ifndef NDEBUG
  for (int i = 0; i < num_constraints_; i++) {
    if (w_.w0(i) < w_.w1.col(i).norm()) {
      throw std::runtime_error("SetW Failed: W not inside cone.");
    }
  }
#endif

  Qsqrt_.resize(num_constraints_);

  for (int i = 0; i < num_constraints_; i++) {

    VectorXd w1 = w_.w1.col(i);
    double w0 = w_.w0(i);
    Sqrt(w1.norm(), &w0, &w1);

    int n = 1 + w1.rows();
    double detw = w0 * w0 - w1.squaredNorm();
    Eigen::MatrixXd W(n, n);
    W.setIdentity();
    W.array() *= detw;
    W(2, 2) *= -1;

    VectorXd wtemp(3);
    wtemp(2) = w0;
    wtemp.head(2) = w1;
    W += 2 * wtemp * wtemp.transpose();
#if USE_SCALED_JORDAN_MULT
    W *= 0.5;
#endif
    Qsqrt_.at(i) = W;
  }

  W_.resize(num_constraints_);
  for (int i = 0; i < num_constraints_; i++) {
    W_.at(i) = (Matrix3d::Identity() +
                Qsqrt_.at(i) * R_.at(i).asDiagonal() * Qsqrt_.at(i))
                   .inverse();
  }

  /* Q W Q :=   Q (I + Q * R * Q)^{-1} Q */
  QWQ_.resize(num_constraints_);
  for (int i = 0; i < num_constraints_; i++) {
    Matrix3d temp;
    temp = Qsqrt_.at(i) * R_.at(i).asDiagonal() * Qsqrt_.at(i);
    temp.diagonal().array() += 1;
    Eigen::LDLT<Eigen::MatrixXd> llt(temp);

#ifndef NDEBUG
    if (llt.info() != Eigen::Success) {
      DUMP(QWQ_.at(i));
      DUMP(temp);
      DUMP(R_.at(i));
      DUMP(Qsqrt_.at(i));
      throw std::runtime_error("NOT PSD!");
    } 
#endif

    QWQ_.at(i) = Qsqrt_.at(i) * llt.solve(Qsqrt_.at(i));
  }

  newton_system_factored_ = false;
}

Eigen::VectorXd SolveNormEquationsPlus(double a, double x0, double x1,
                                       double y0, double y1, double k) {
  Eigen::VectorXd t(2);
  double a_squared = a * a;
  double under_radical = a_squared * y1 + 2 * a * k * y0 - 2 * a * x0 * y1 +
                         k * k - 2 * k * x0 * y0 + x0 * x0 * y1 + x1 * y0 * y0 -
                         x1 * y1;

  if (under_radical > 1e-16) {
    t(0) = (-sqrt(under_radical) + a * y0 + k - x0 * y0) / (y0 * y0 - y1);
    t(1) = (sqrt(under_radical) + a * y0 + k - x0 * y0) / (y0 * y0 - y1);
  } else {
    if (under_radical >= 0) {
      t.resize(1);
      t(0) = (a * y0 + k - x0 * y0) / (y0 * y0 - y1);
    } else {
      t.resize(0);
    }
  }

  return t;
}

Eigen::VectorXd SolveNormEquationsMinus(double a, double x0, double x1,
                                        double y0, double y1, double k) {
  Eigen::VectorXd t(2);
  double a_squared = a * a;
  double under_radical = a_squared * y1 + 2 * a * k * y0 - 2 * a * x0 * y1 +
                         k * k - 2 * k * x0 * y0 + x0 * x0 * y1 + x1 * y0 * y0 -
                         x1 * y1;

  if (under_radical > 1e-16) {
    t(0) = (-sqrt(under_radical) + a * y0 + k - x0 * y0) / (y0 * y0 - y1 + 1e-15);
    t(1) = (sqrt(under_radical) + a * y0 + k - x0 * y0) / (y0 * y0 - y1 + 1e-15);
  } else {
    if (under_radical >= 0) {
      t.resize(1);
      t(0) = (a * y0 + k - x0 * y0) / (y0 * y0 - y1 + 1e-15);
    } else {
      t.resize(0);
    }
  }

  return t;
}

Eigen::VectorXd GetCandidateK(double dinfmax, double x0, double x1, double y0,
                              double y1, double k) {
  auto t = SolveNormEquationsPlus(dinfmax, x0, x1, y0, y1, k);
  std::vector<double> val;
  double eps = 0.01;
  for (int i = 0; i < t.size(); i++) {
    double error_minus =
        x0 + t(i) * y0 - sqrt(x1 + 2 * t(i) * k + t(i) * t(i) * y1) + dinfmax;
    double error_plus =
        x0 + t(i) * y0 + sqrt(x1 + 2 * t(i) * k + t(i) * t(i) * y1) - dinfmax;

    if ((fabs(error_plus) < eps && error_minus > -eps) ||
       (fabs(error_minus) < eps && error_plus < eps) )  {
      val.push_back(t(i));
    } else {
      //DUMP(error_minus);
      //DUMP(error_plus);
    }
  }
  t = SolveNormEquationsPlus(-dinfmax, x0, x1, y0, y1, k);
  for (int i = 0; i < t.size(); i++) {
    double error_minus =
        x0 + t(i) * y0 - sqrt(x1 + 2 * t(i) * k + t(i) * t(i) * y1) + dinfmax;
    double error_plus =
        x0 + t(i) * y0 + sqrt(x1 + 2 * t(i) * k + t(i) * t(i) * y1) - dinfmax;

    if ((fabs(error_plus) < eps && error_minus > -eps) ||
       (fabs(error_minus) < eps && error_plus < eps) )  {
      val.push_back(t(i));
    } else {
      //DUMP(error_minus);
      //DUMP(error_plus);
    }
  }
  return Eigen::Map<const VectorXd>(val.data(), static_cast<int>(val.size()));
}

double GetMinSqrtMu(double dinfmax_unscaled, const SpinFactorProduct& x,
                    const SpinFactorProduct& y) {

#if USE_SCALED_JORDAN_MULT
  // We find 1/sqrt(2) * (x_0 \pm x_1) = dinf_max_unscaled
  double dinfmax = dinfmax_unscaled * std::sqrt(2.0);
#else
  double dinfmax = dinfmax_unscaled;
#endif

  double upper_bound = 1e45;
  double lower_bound = -1e45;
  for (int i = 0; i < x.w0.size(); i++) {
    auto t =
        GetCandidateK(dinfmax, x.w0(i), x.w1.col(i).squaredNorm(), y.w0(i),
                      y.w1.col(i).squaredNorm(), x.w1.col(i).dot(y.w1.col(i)));

    if (t.size() < 2) {
      return -1;
    }

    double lower_bound_i = t.minCoeff();
    double upper_bound_i = t.maxCoeff();

    if (lower_bound_i > lower_bound) {
      lower_bound = lower_bound_i;
    }
    if (upper_bound_i < upper_bound) {
      upper_bound = upper_bound_i;
    }

#ifndef NDEBUG
    for (int j = 0; j < t.size(); j++) {
      double v0 = x.w0(i) + t(j) * y.w0(i);
      VectorXd v1 = x.w1.col(i) + t(j) * y.w1.col(i);
      double val = fabs(v0 + v1.norm());
      if (fabs(v0 - v1.norm()) > val) {
        val = fabs(v0 - v1.norm());
      }
      if (fabs(val - dinfmax) > line_search_tolerance) {
        return -1;
        throw std::runtime_error("Bad calculation 1.");
      }
    }
#endif
  }




  if (lower_bound > upper_bound) {
    return -1;
  }

#ifndef NDEBUG
  VectorXd d = Vect(x) + lower_bound * Vect(y);
  int num_contact = x.w0.size();
  double test0 = NormInf(SpinFactorProduct{d, num_contact});

  d = Vect(x) + upper_bound * Vect(y);
  double test1 = NormInf(SpinFactorProduct{d, num_contact});
  if (fabs(test0 - dinfmax_unscaled) > line_search_tolerance) {
    //throw std::runtime_error("Bad calculation 2.");
  }
  if (fabs(test1 - dinfmax_unscaled) > line_search_tolerance) {
    //throw std::runtime_error("Bad calculation 3.");
  }
#endif


  return upper_bound;
}

/*
  for (int i = 0; i < num_contacts; i++) {
    SpinFactorElement invw;
    invw.w0 = w0.w0(i) + 1e-5; invw.w1 = w0.w1.col(i);
    Inverse(invw.w1.norm(), &invw.w0, &invw.w1);

    w.w0(i) = invw.w0;
    w.w1.col(i) = invw.w1;
  }
*/

GeodesicSolverSolution GeodesicSolver(
    const GeodesicSolverSolution& sol,
    const std::vector<conex::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat,
    const GeodesicSolverOptions& options) {
  const NewtonDirection dir(J, M, num_block_rows_of_J, R, num_contacts);

  const VectorXd slack =
      dir.J_.Multiply(sol.v) - v_hat + R.asDiagonal() * sol.lambda;

  int offset = 0;
  VectorXd w0_vect = sol.lambda;
  for (int i = 0; i < num_contacts; i++) {
    w0_vect(offset + 2) += 1e-6;
    offset += 3;
  }

  double mu = slack.dot(w0_vect) / num_contacts;
  double eps = 1e-9;
  if (mu < eps) {
    mu = eps;
  }

  w0_vect = w0_vect * 1.0 / std::sqrt(mu);
  SpinFactorProduct w0(w0_vect, num_contacts);

  return GeodesicSolver(w0, J, M, R, num_block_rows_of_J, num_contacts, vstar,
                        v_hat, options);
}


GeodesicSolverSolution GeodesicSolver(
    const GeodesicSolverSolution& sol,
    const VectorXd& friction_coefficient,
    const std::vector<conex::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat,
    const GeodesicSolverOptions& options) {

  SparseJacobian Jscale(J);
  Jscale.RescaleByFrictionCoefficient(friction_coefficient);
  auto sol_scale = sol;

  auto Rscaled = R;
  auto vc_stab_scaled = v_hat;

  int offset = 0;
  for (int i = 0; i < num_contacts; i++) {
    Rscaled(offset + 2) *=  1.0/(friction_coefficient(i)  * friction_coefficient(i));
    vc_stab_scaled(offset + 2) *=  1.0/(friction_coefficient(i));
    sol_scale.lambda(offset + 2) *= friction_coefficient(i) ;
    offset += 3;
  }


  auto solution = GeodesicSolver(sol_scale,
  Jscale.blocks(),
  M, 
  Rscaled,
  num_block_rows_of_J, num_contacts, 
  vstar,
  vc_stab_scaled,
  options);

  offset = 0;
  for (int i = 0; i < num_contacts; i++) {
    solution.lambda(offset + 2) /= friction_coefficient(i);
    offset += 3;
  }
  return solution;
}






GeodesicSolverSolution GeodesicSolver(
    const SpinFactorProduct& w0,
    const std::vector<conex::BlockMatrixTriplet>& J,
    const std::vector<Eigen::MatrixXd>& M, const Eigen::VectorXd& R,
    int num_block_rows_of_J, int num_contacts, const Eigen::VectorXd& vstar,
    const Eigen::VectorXd& v_hat,
    const GeodesicSolverOptions& options) {

  SpinFactorProduct w = w0;

  double identity_scale = 100;
  double min_mu = options.target_mu;

  NewtonDirection dir(J, M, num_block_rows_of_J, R, num_contacts);

  Eigen::VectorXd d;
  Eigen::VectorXd v;
  Eigen::VectorXd v0;
  Eigen::VectorXd vu;

  Eigen::VectorXd d0;
  Eigen::VectorXd du;
  Eigen::VectorXd dt; 

  double inv_sqrt_mu = (-(du - d0).dot(d0) / (d0 - du).squaredNorm());
  if (inv_sqrt_mu < 0) {
    inv_sqrt_mu = 1e-8;
  }

  double mu = .1;
  double dinf = 0;

  GeodesicSolverSolution sol;
  bool decrease_mu_on_fail = false;

  double dinf_max = 1.01;
  double inv_sqrt_mu_hat;
  int i = 0;
  int max_iterations = options.maximum_iterations;

  for (; i < max_iterations; i++) {
    dir.SetW(w);
    dir.CalculateNewtonDirection(0, vstar, v_hat, &d0, &v0);
    dir.CalculateNewtonDirection(1, vstar, v_hat, &du, &vu);

    dt = du - d0;
    inv_sqrt_mu_hat = std::fabs(d0.dot(dt) / (dt).squaredNorm());
    if (i == 0) {
      inv_sqrt_mu = inv_sqrt_mu_hat;
    } else {
      inv_sqrt_mu = GetMinSqrtMu(dinf_max, SpinFactorProduct(d0, num_contacts),
                                 SpinFactorProduct(dt, num_contacts));
    }
    

    if (inv_sqrt_mu < 0) {
      if (!decrease_mu_on_fail || dinf > 2) {
        inv_sqrt_mu = inv_sqrt_mu_hat;
      } else {
        inv_sqrt_mu = 2.5 * sqrt(1.0 / mu);
      }
    }

    if (inv_sqrt_mu > 1.0 / std::sqrt(1e-9)) {
      inv_sqrt_mu = 1.0 / std::sqrt(1e-9);
    }

    d = d0 + inv_sqrt_mu*(du-d0);
    v = v0 + inv_sqrt_mu*(vu-v0);
    dinf = NormInf(SpinFactorProduct{d, num_contacts});
    double dnorm = d.norm();
    double scale = 2.0 / (dinf * dinf);

    if (scale < 1) {
      d.array() *= scale;
    } else {
      scale = 1;
    }

    mu = 1.0 / (inv_sqrt_mu * inv_sqrt_mu);
    double abort_warmstart = i == 0 && (dinf > 1.05 || mu > 100);
    if (options.verbosity) {
      std::cout << std::setprecision(2);
      std::cout << std::scientific;
      if (abort_warmstart) {
        std::cout << "*";
      } else {
        std::cout << " ";
      }
      std::cout << "iter: " << i << "  mu: " << mu << " |d|_inf " << dinf 
                << " mu_hat " << 1.0/(inv_sqrt_mu_hat*inv_sqrt_mu_hat)
                << "  |d| " << dnorm << "  stepsize: " << scale << "\n";
    }

    if (abort_warmstart) {
      w.w0.setConstant(identity_scale);
      w.w1.setConstant(0);
      continue;
    }

    SpinFactorProduct newton_dir(d, num_contacts);

    if (mu >= 2*min_mu) {
      dinf_max = 1.1 * sqrt(2);
    } else {
      dinf_max = .99;
    }

    sol.info.iterations = i + 1;
    if (dinf <= 1 + line_search_tolerance) {
      if (mu <= min_mu + 1e-9) {
        TakeStep(w, newton_dir, &w, true);
        break;
      }
    }
    TakeStep(w, newton_dir, &w, false);
    decrease_mu_on_fail = true;
  }
  sol.v = v * 1.0 / inv_sqrt_mu;
  sol.lambda = Vect(w) * 1.0 / inv_sqrt_mu;
  sol.info.failed = i >= max_iterations;
  return sol;
}
