/*-------------------------------------------------------------------------------
  Copyright (c) 2024 GRF Contributors.

  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_LOCALPOLYNOMIALREGRESSION_H
#define GRF_LOCALPOLYNOMIALREGRESSION_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "Eigen/Dense"

namespace grf {

enum class KernelType {
  TRIANGULAR = 0,
  UNIFORM = 1,
  EPANECHNIKOV = 2
};

inline double kernel_weight(KernelType kernel, double u_abs) {
  switch (kernel) {
    case KernelType::TRIANGULAR:
      return std::max(0.0, 1.0 - u_abs);
    case KernelType::UNIFORM:
      return (u_abs <= 1.0) ? 1.0 : 0.0;
    case KernelType::EPANECHNIKOV: {
      if (u_abs > 1.0) return 0.0;
      double u2 = u_abs * u_abs;
      return 0.75 * (1.0 - u2);
    }
    default:
      return 0.0;
  }
}

// Fits a one-sided local polynomial regression at 0 and returns the intercept.
// The caller supplies accessors for running_var, response, and base weights.
template <typename RunningGetter, typename ResponseGetter, typename WeightGetter>
inline bool local_polynomial_intercept_at_zero_one_sided(
    const std::vector<size_t>& samples,
    RunningGetter running_var,
    ResponseGetter response,
    WeightGetter weight,
    bool right_side,
    int order,
    double bandwidth,
    KernelType kernel,
    double& intercept_out) {

  if (bandwidth <= 0) {
    return false;
  }
  if (order < 0) {
    return false;
  }

  const int p = order + 1;
  Eigen::MatrixXd xtwx = Eigen::MatrixXd::Zero(p, p);
  Eigen::VectorXd xtwy = Eigen::VectorXd::Zero(p);

  double total_weight = 0.0;

  for (size_t sample : samples) {
    double r = running_var(sample);
    if (right_side) {
      if (r < 0) continue;
    } else {
      if (r >= 0) continue;
    }

    double u_abs = std::abs(r) / bandwidth;
    double k = kernel_weight(kernel, u_abs);
    if (k <= 0) continue;

    double w = weight(sample) * k;
    if (!(w > 0)) continue;

    Eigen::VectorXd x(p);
    x(0) = 1.0;
    for (int j = 1; j < p; ++j) {
      x(j) = x(j - 1) * r;
    }

    xtwx.noalias() += w * (x * x.transpose());
    xtwy.noalias() += w * x * response(sample);
    total_weight += w;
  }

  if (total_weight <= 0) {
    return false;
  }

  Eigen::LDLT<Eigen::MatrixXd> ldlt(xtwx);
  if (ldlt.info() != Eigen::Success) {
    return false;
  }
  Eigen::VectorXd beta = ldlt.solve(xtwy);
  if (ldlt.info() != Eigen::Success) {
    return false;
  }

  intercept_out = beta(0);
  return std::isfinite(intercept_out);
}

} // namespace grf

#endif // GRF_LOCALPOLYNOMIALREGRESSION_H

