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

#include <cmath>
#include <vector>

#include "commons/Data.h"
#include "commons/utility.h"
#include "commons/LocalPolynomialRegression.h"
#include "prediction/RDPredictionStrategy.h"

namespace grf {

const std::size_t RDPredictionStrategy::OUTCOME = 0;
const std::size_t RDPredictionStrategy::TREATMENT = 1;
const std::size_t RDPredictionStrategy::INSTRUMENT = 2;
const std::size_t RDPredictionStrategy::OUTCOME_INSTRUMENT = 3;
const std::size_t RDPredictionStrategy::TREATMENT_INSTRUMENT = 4;
const std::size_t RDPredictionStrategy::INSTRUMENT_INSTRUMENT = 5;
const std::size_t RDPredictionStrategy::WEIGHT = 6;
const std::size_t RDPredictionStrategy::NUM_TYPES = 7;

size_t RDPredictionStrategy::prediction_length() const {
    return 1;
}

std::vector<double> RDPredictionStrategy::predict(const std::vector<double>& average) const {
  double instrument_effect_numerator = average.at(OUTCOME_INSTRUMENT) * average.at(WEIGHT)
    - average.at(OUTCOME) * average.at(INSTRUMENT);
  double first_stage_numerator = average.at(TREATMENT_INSTRUMENT) * average.at(WEIGHT)
    - average.at(TREATMENT) * average.at(INSTRUMENT);

  return { instrument_effect_numerator / first_stage_numerator };
}

std::vector<double> RDPredictionStrategy::compute_variance(
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    size_t ci_group_size) const {

  double instrument_effect_numerator = average.at(OUTCOME_INSTRUMENT) * average.at(WEIGHT)
     - average.at(OUTCOME) * average.at(INSTRUMENT);
  double first_stage_numerator = average.at(TREATMENT_INSTRUMENT) * average.at(WEIGHT)
     - average.at(TREATMENT) * average.at(INSTRUMENT);
  double treatment_effect_estimate = instrument_effect_numerator / first_stage_numerator;
  double main_effect_estimate = (average.at(OUTCOME) - average.at(TREATMENT) * treatment_effect_estimate)
     / average.at(WEIGHT);

  double num_good_groups = 0;
  double rho_squared = 0;
  double rho_grouped_squared = 0;

  for (size_t group = 0; group < leaf_values.get_num_nodes() / ci_group_size; ++group) {
    bool good_group = true;
    for (size_t j = 0; j < ci_group_size; ++j) {
      if (leaf_values.empty(group * ci_group_size + j)) {
        good_group = false;
      }
    }
    if (!good_group) continue;

    num_good_groups++;

    double group_rho = 0;

    for (size_t j = 0; j < ci_group_size; ++j) {

      size_t i = group * ci_group_size + j;
      const std::vector<double>& leaf_value = leaf_values.get_values(i);

      double psi_1 = leaf_value.at(OUTCOME_INSTRUMENT)
                     - leaf_value.at(TREATMENT_INSTRUMENT) * treatment_effect_estimate
                     - leaf_value.at(INSTRUMENT) * main_effect_estimate;
      double psi_2 = leaf_value.at(OUTCOME)
                     - leaf_value.at(TREATMENT) * treatment_effect_estimate
                     - leaf_value.at(WEIGHT) * main_effect_estimate;

      double rho = (psi_1 * average.at(WEIGHT) - psi_2 * average.at(INSTRUMENT)) / first_stage_numerator;
      group_rho += rho;
      rho_squared += rho * rho;
    }

    group_rho /= ci_group_size;
    rho_grouped_squared += group_rho * group_rho;
  }

  double var_between = rho_grouped_squared / num_good_groups;
  double var_total = rho_squared / (num_good_groups * ci_group_size);
  double group_noise = (var_total - var_between) / (ci_group_size - 1);
  double var_debiased = bayes_debiaser.debias(var_between, group_noise, num_good_groups);

  return { var_debiased };
}

size_t RDPredictionStrategy::prediction_value_length() const {
  return NUM_TYPES;
}

PredictionValues RDPredictionStrategy::precompute_prediction_values(
    const std::vector<std::vector<size_t>>& leaf_samples,
    const Data& data) const {
  size_t num_leaves = leaf_samples.size();

  std::vector<std::vector<double>> values(num_leaves);

  for (size_t i = 0; i < leaf_samples.size(); ++i) {
    size_t leaf_size = leaf_samples[i].size();
    if (leaf_size == 0) {
      continue;
    }

    double sum_Y = 0;
    double sum_W = 0;
    double sum_Z = 0;
    double sum_YZ = 0;
    double sum_WZ = 0;
    double sum_ZZ = 0;

    double sum_weight = 0.0;
    for (auto& sample : leaf_samples[i]) {
      auto weight = data.get_weight(sample);
      sum_Y += weight * data.get_outcome(sample);
      sum_W += weight * data.get_treatment(sample);
      sum_Z += weight * data.get_instrument(sample);
      sum_YZ += weight * data.get_outcome(sample) * data.get_instrument(sample);
      sum_WZ += weight * data.get_treatment(sample) * data.get_instrument(sample);
      sum_ZZ += weight * data.get_instrument(sample) * data.get_instrument(sample);
      sum_weight += weight;
    }

    // if total weight is very small, treat the leaf as empty
    if (std::abs(sum_weight) <= 1e-16) {
      continue;
    }
    std::vector<double>& value = values[i];
    value.resize(NUM_TYPES);

    value[OUTCOME] = sum_Y / leaf_size;
    value[TREATMENT] = sum_W / leaf_size;
    value[INSTRUMENT] = sum_Z / leaf_size;
    value[OUTCOME_INSTRUMENT] = sum_YZ / leaf_size;
    value[TREATMENT_INSTRUMENT] = sum_WZ / leaf_size;
    value[INSTRUMENT_INSTRUMENT] = sum_ZZ / leaf_size;
    value[WEIGHT] = sum_weight / leaf_size;
  }

  return PredictionValues(values, NUM_TYPES);
}

std::vector<std::pair<double, double>> RDPredictionStrategy::compute_error(
    size_t sample,
    const std::vector<double>& average,
    const PredictionValues& leaf_values,
    const Data& data) const {

  double reduced_form_numerator = average.at(OUTCOME_INSTRUMENT) * average.at(WEIGHT)
    - average.at(OUTCOME) * average.at(INSTRUMENT);
  double reduced_form_denominator = average.at(INSTRUMENT_INSTRUMENT) * average.at(WEIGHT)
    - average.at(INSTRUMENT) * average.at(INSTRUMENT);
  double reduced_form_estimate = reduced_form_numerator / reduced_form_denominator;

  double outcome = data.get_outcome(sample);
  double instrument = data.get_instrument(sample);

  double residual = outcome - (instrument - average.at(INSTRUMENT) / average.at(WEIGHT)) * reduced_form_estimate - average.at(OUTCOME) / average.at(WEIGHT);
  double error_raw = residual * residual;

  size_t num_trees = 0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    num_trees++;
  }

  if (num_trees <= 5) {
    return { std::make_pair<double, double>(NAN, NAN) };
  }

  double error_bias = 0.0;
  for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
    if (leaf_values.empty(n)) {
      continue;
    }
    const std::vector<double>& leaf_value = leaf_values.get_values(n);
    double weight_loto = (num_trees * average.at(WEIGHT) - leaf_value.at(WEIGHT)) / (num_trees - 1);
    double outcome_loto = (num_trees * average.at(OUTCOME) - leaf_value.at(OUTCOME)) / (num_trees - 1);
    double instrument_loto = (num_trees * average.at(INSTRUMENT) - leaf_value.at(INSTRUMENT)) / (num_trees - 1);
    double outcome_instrument_loto = (num_trees * average.at(OUTCOME_INSTRUMENT) - leaf_value.at(OUTCOME_INSTRUMENT)) / (num_trees - 1);
    double instrument_instrument_loto = (num_trees * average.at(INSTRUMENT_INSTRUMENT) - leaf_value.at(INSTRUMENT_INSTRUMENT)) / (num_trees - 1);

    double reduced_form_numerator_loto = outcome_instrument_loto * weight_loto - outcome_loto * instrument_loto;
    double reduced_form_denominator_loto = instrument_instrument_loto * weight_loto - instrument_loto * instrument_loto;
    double reduced_form_estimate_loto = reduced_form_numerator_loto / reduced_form_denominator_loto;

    double residual_loto = outcome - (instrument - instrument_loto / weight_loto) * reduced_form_estimate_loto - outcome_loto / weight_loto;
    error_bias += (residual_loto - residual) * (residual_loto - residual);
  }

  error_bias *= ((double) (num_trees - 1)) / num_trees;

  double debiased_error = error_raw - error_bias;

  auto output = std::make_pair(debiased_error, error_bias);
  return {output};

}

} // namespace grf
