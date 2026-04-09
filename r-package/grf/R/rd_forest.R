#' Regression discontinuity forest
#'
#' Trains a regression discontinuity forest that can be used to estimate
#' conditional treatment effects identified using a cutoff-based assignment rule.
#' This forest is implemented as an instrumental forest where the instrument is
#' the cutoff indicator \eqn{Z = 1\{running \ge cutoff\}}.
#'
#' @param X The covariates used in the instrumental regression.
#' @param Y The outcome.
#' @param W The treatment received (may be binary or real).
#' @param running The running variable used to define the cutoff assignment.
#' @param cutoff Treatment threshold applied to the running variable. Default is 0.
#' @param Y.hat Estimates of the expected responses E[Y | Xi], marginalizing
#'              over treatment. If Y.hat = NULL, these are estimated using
#'              a separate regression forest. Default is NULL.
#' @param W.hat Estimates of the treatment propensities E[W | Xi]. If W.hat = NULL,
#'              these are estimated using a separate regression forest. Default is NULL.
#' @param Z.hat Estimates of the instrument propensities E[Z | Xi]. If Z.hat = NULL,
#'              these are estimated using a separate regression forest. Default is NULL.
#' @param num.trees Number of trees grown in the forest. Default is 2000.
#' @param sample.weights Weights given to each observation in estimation.
#'                       If NULL, each observation receives equal weight. Default is NULL.
#' @param clusters Vector of integers or factors specifying which cluster each observation corresponds to.
#'  Default is NULL (ignored).
#' @param equalize.cluster.weights If FALSE, each unit is given the same weight (so that bigger
#'  clusters get more weight). If TRUE, each cluster is given equal weight in the forest.
#'  Default is FALSE.
#' @param sample.fraction Fraction of the data used to build each tree. Default is 0.5.
#' @param mtry Number of variables tried for each split. Default is sqrt(p) + 20.
#' @param min.node.size A target for the minimum number of observations in each tree leaf. Default is 5.
#' @param honesty Whether to use honest splitting. Default is TRUE.
#' @param honesty.fraction Fraction of data used for determining splits if honesty = TRUE. Default is 0.5.
#' @param honesty.prune.leaves Whether to prune empty leaves when honesty is TRUE. Default is TRUE.
#' @param alpha Maximum imbalance of a split. Default is 0.05.
#' @param imbalance.penalty Penalty for imbalanced splits. Default is 0.
#' @param stabilize.splits Whether or not the instrument should be taken into account when
#'                         determining the imbalance of a split. Default is TRUE.
#' @param ci.group.size The forest will grow ci.group.size trees on each subsample. Default is 2.
#' @param reduced.form.weight Whether splits should be regularized towards a naive
#'                            splitting criterion that ignores the instrument (and
#'                            instead emulates a causal forest). Default is 0.
#' @param tune.parameters Whether/how to tune parameters (see other forest types). Default is "none".
#' @param tune.num.trees Number of trees used for tuning. Default is 200.
#' @param tune.num.reps Number of random tuning draws. Default is 50.
#' @param tune.num.draws Number of subsamples used for tuning. Default is 1000.
#' @param compute.oob.predictions Whether OOB predictions on training set should be precomputed. Default is TRUE.
#' @param num.threads Number of threads used in training. Default is NULL.
#' @param seed Random seed. Default is a random draw.
#'
#' @return A trained rd_forest object.
#'
#' @export
rd_forest <- function(X, Y, W, running, cutoff = 0,
                      Y.hat = NULL,
                      W.hat = NULL,
                      Z.hat = NULL,
                      num.trees = 2000,
                      sample.weights = NULL,
                      clusters = NULL,
                      equalize.cluster.weights = FALSE,
                      sample.fraction = 0.5,
                      mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                      min.node.size = 5,
                      honesty = TRUE,
                      honesty.fraction = 0.5,
                      honesty.prune.leaves = TRUE,
                      alpha = 0.05,
                      imbalance.penalty = 0,
                      stabilize.splits = TRUE,
                      ci.group.size = 2,
                      reduced.form.weight = 0,
                      tune.parameters = "none",
                      tune.num.trees = 200,
                      tune.num.reps = 50,
                      tune.num.draws = 1000,
                      compute.oob.predictions = TRUE,
                      num.threads = NULL,
                      seed = runif(1, 0, .Machine$integer.max)) {
  has.missing.values <- validate_X(X, allow.na = TRUE)
  validate_sample_weights(sample.weights, X)
  Y <- validate_observations(Y, X)
  W <- validate_observations(W, X)
  running <- validate_observations(running, X)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, sample.weights)
  num.threads <- validate_num_threads(num.threads)

  if (!is.numeric(cutoff) || length(cutoff) != 1) {
    stop("cutoff must be a numeric scalar.")
  }

  # Center the running variable so the threshold is always at 0.
  running.orig <- running
  running <- running.orig - cutoff
  Z <- as.numeric(running >= 0)

  if (!is.numeric(reduced.form.weight) | reduced.form.weight < 0 | reduced.form.weight > 1) {
    stop("Error: Invalid value for reduced.form.weight. Please give a value in [0,1].")
  }

  all.tunable.params <- c("sample.fraction", "mtry", "min.node.size", "honesty.fraction",
                          "honesty.prune.leaves", "alpha", "imbalance.penalty")
  default.parameters <- list(sample.fraction = 0.5,
                             mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                             min.node.size = 5,
                             honesty.fraction = 0.5,
                             honesty.prune.leaves = TRUE,
                             alpha = 0.05,
                             imbalance.penalty = 0)

  # Match causal_forest orthogonalization defaults to ensure rd_forest behaves
  # the same as causal_forest when W coincides with the cutoff assignment Z.
  args.orthog = list(X = X,
                     num.trees = max(50, num.trees / 4),
                     sample.weights = sample.weights,
                     clusters = clusters,
                     equalize.cluster.weights = equalize.cluster.weights,
                     sample.fraction = sample.fraction,
                     mtry = mtry,
                     min.node.size = 5,
                     honesty = TRUE,
                     honesty.fraction = 0.5,
                     honesty.prune.leaves = honesty.prune.leaves,
                     alpha = alpha,
                     imbalance.penalty = imbalance.penalty,
                     ci.group.size = 1,
                     tune.parameters = tune.parameters,
                     num.threads = num.threads,
                     seed = seed)

  if (is.null(Y.hat)) {
    forest.Y <- do.call(regression_forest, c(Y = list(Y), args.orthog))
    Y.hat <- predict(forest.Y)$predictions
  } else if (length(Y.hat) == 1) {
    Y.hat <- rep(Y.hat, nrow(X))
  } else if (length(Y.hat) != nrow(X)) {
    stop("Y.hat has incorrect length.")
  }

  if (is.null(W.hat)) {
    forest.W <- do.call(regression_forest, c(Y = list(W), args.orthog))
    W.hat <- predict(forest.W)$predictions
  } else if (length(W.hat) == 1) {
    W.hat <- rep(W.hat, nrow(X))
  } else if (length(W.hat) != nrow(X)) {
    stop("W.hat has incorrect length.")
  }

  if (is.null(Z.hat)) {
    forest.Z <- do.call(regression_forest, c(Y = list(Z), args.orthog))
    Z.hat <- predict(forest.Z)$predictions
  } else if (length(Z.hat) == 1) {
    Z.hat <- rep(Z.hat, nrow(X))
  } else if (length(Z.hat) != nrow(X)) {
    stop("Z.hat has incorrect length.")
  }

  data <- create_train_matrices(X, outcome = Y - Y.hat, treatment = W - W.hat,
                                instrument = Z - Z.hat, running_var = running,
                                sample.weights = sample.weights)
  args <- list(num.trees = num.trees,
              clusters = clusters,
              samples.per.cluster = samples.per.cluster,
              sample.fraction = sample.fraction,
              mtry = mtry,
              min.node.size = min.node.size,
              honesty = honesty,
              honesty.fraction = honesty.fraction,
              honesty.prune.leaves = honesty.prune.leaves,
              alpha = alpha,
              imbalance.penalty = imbalance.penalty,
              stabilize.splits = stabilize.splits,
              ci.group.size = ci.group.size,
              reduced.form.weight = reduced.form.weight,
              compute.oob.predictions = compute.oob.predictions,
              num.threads = num.threads,
              seed = seed,
              legacy.seed = get_legacy_seed(),
              verbose = get_verbose())

  tuning.output <- NULL
  if (!identical(tune.parameters, "none")) {
    if (identical(tune.parameters, "all")) {
      tune.parameters <- all.tunable.params
    } else {
      tune.parameters <- unique(match.arg(tune.parameters, all.tunable.params, several.ok = TRUE))
    }
    if (!honesty) {
      tune.parameters <- tune.parameters[!grepl("honesty", tune.parameters)]
    }
    tune.parameters.defaults <- default.parameters[tune.parameters]
    tuning.output <- tune_forest(data = data,
                                 nrow.X = nrow(X),
                                 ncol.X = ncol(X),
                                 args = args,
                                 tune.parameters = tune.parameters,
                                 tune.parameters.defaults = tune.parameters.defaults,
                                 tune.num.trees = tune.num.trees,
                                 tune.num.reps = tune.num.reps,
                                 tune.num.draws = tune.num.draws,
                                 train = rd_train)

    args <- utils::modifyList(args, as.list(tuning.output[["params"]]))
  }

  forest <- do.call.rcpp(rd_train, c(data, args))
  class(forest) <- c("rd_forest", "grf")
  forest[["seed"]] <- seed
  forest[["num.threads"]] <- num.threads
  forest[["ci.group.size"]] <- ci.group.size
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["W.orig"]] <- W
  forest[["running.orig"]] <- running.orig
  forest[["running.adjusted"]] <- running
  forest[["cutoff"]] <- cutoff
  forest[["Z.orig"]] <- Z
  forest[["Y.hat"]] <- Y.hat
  forest[["W.hat"]] <- W.hat
  forest[["Z.hat"]] <- Z.hat
  forest[["clusters"]] <- clusters
  forest[["equalize.cluster.weights"]] <- equalize.cluster.weights
  forest[["sample.weights"]] <- sample.weights
  forest[["tunable.params"]] <- args[all.tunable.params]
  forest[["tuning.output"]] <- tuning.output
  forest[["has.missing.values"]] <- has.missing.values

  forest
}

#' Predict with a regression discontinuity forest
#'
#' Gets estimates of tau(x) using a trained regression discontinuity forest.
#'
#' @param object The trained forest.
#' @param newdata Points at which predictions should be made. If NULL, makes out-of-bag
#'                predictions on the training set instead.
#' @param num.threads Number of threads used in prediction. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param estimate.variance Whether variance estimates for \eqn{\hat\\tau(x)} are desired.
#' @param ... Additional arguments (currently ignored).
#'
#' @return Vector of predictions, along with (optional) variance estimates.
#'
#' @method predict rd_forest
#' @export
predict.rd_forest <- function(object, newdata = NULL,
                              num.threads = NULL,
                              estimate.variance = FALSE,
                              ...) {

  if (is.null(newdata) && !estimate.variance && !is.null(object$predictions)) {
    return(data.frame(
      predictions = object$predictions,
      debiased.error = object$debiased.error
    ))
  }

  num.threads <- validate_num_threads(num.threads)
  forest.short <- object[-which(names(object) == "X.orig")]

  X <- object[["X.orig"]]
  Y.centered <- object[["Y.orig"]] - object[["Y.hat"]]
  W.centered <- object[["W.orig"]] - object[["W.hat"]]
  Z.centered <- object[["Z.orig"]] - object[["Z.hat"]]
  running.centered <- object[["running.adjusted"]]

  train.data <- create_train_matrices(X, outcome = Y.centered, treatment = W.centered,
                                      instrument = Z.centered, running_var = running.centered)
  args <- list(forest.object = forest.short,
               num.threads = num.threads,
               estimate.variance = estimate.variance,
               verbose = get_verbose())

  if (!is.null(newdata)) {
    validate_newdata(newdata, X, allow.na = TRUE)
    test.data <- create_test_matrices(newdata)
    ret <- do.call.rcpp(rd_predict, c(train.data, test.data, args))
  } else {
    ret <- do.call.rcpp(rd_predict_oob, c(train.data, args))
  }

  empty <- sapply(ret, function(elem) length(elem) == 0)
  do.call(cbind.data.frame, ret[!empty])
}
