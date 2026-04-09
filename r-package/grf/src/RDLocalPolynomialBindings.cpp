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

#include <Rcpp.h>
#include <vector>

#include "commons/LocalPolynomialRegression.h"

using namespace grf;

static KernelType parse_kernel(const std::string& kernel) {
  if (kernel == "triangular") return KernelType::TRIANGULAR;
  if (kernel == "uniform") return KernelType::UNIFORM;
  if (kernel == "epanechnikov") return KernelType::EPANECHNIKOV;
  throw std::runtime_error("Unknown kernel: " + kernel);
}

// [[Rcpp::export]]
double rd_local_polynomial_tau(const Rcpp::NumericVector& running_var,
                               const Rcpp::NumericVector& y,
                               int order,
                               double bandwidth,
                               const std::string& kernel,
                               Rcpp::Nullable<Rcpp::NumericVector> weights) {
  if (running_var.size() != y.size()) {
    throw std::runtime_error("running_var and y must have the same length.");
  }
  if (running_var.size() == 0) {
    return NA_REAL;
  }

  KernelType ktype = parse_kernel(kernel);

  const double* rv_ptr = running_var.begin();
  const double* y_ptr = y.begin();

  bool has_weights = weights.isNotNull();
  Rcpp::NumericVector w_vec;
  const double* w_ptr = nullptr;
  if (has_weights) {
    w_vec = Rcpp::NumericVector(weights);
    if (w_vec.size() != running_var.size()) {
      throw std::runtime_error("weights must have the same length as running_var.");
    }
    w_ptr = w_vec.begin();
  }

  std::vector<size_t> samples(static_cast<size_t>(running_var.size()));
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = i;
  }

  auto running_getter = [&](size_t i) { return rv_ptr[i]; };
  auto response_getter = [&](size_t i) { return y_ptr[i]; };
  auto weight_getter = [&](size_t i) { return has_weights ? w_ptr[i] : 1.0; };

  double mu_left = NAN;
  double mu_right = NAN;
  bool ok_left = local_polynomial_intercept_at_zero_one_sided(
      samples, running_getter, response_getter, weight_getter,
      /*right_side*/ false, order, bandwidth, ktype, mu_left);
  bool ok_right = local_polynomial_intercept_at_zero_one_sided(
      samples, running_getter, response_getter, weight_getter,
      /*right_side*/ true, order, bandwidth, ktype, mu_right);

  if (!ok_left || !ok_right) {
    return NA_REAL;
  }
  return mu_right - mu_left;
}

