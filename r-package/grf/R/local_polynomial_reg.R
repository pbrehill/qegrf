#' Local polynomial regression at 0
#'
#' Fits a one-sided local polynomial regression (left/right of 0) and returns the
#' implied discontinuity at 0. This is a small utility intended for RD forests.
#'
#' @param running_var Numeric vector of (centered) running variable values.
#'                    The threshold is assumed to be 0.
#' @param Y Numeric vector of outcomes.
#' @param order Polynomial order. Default is 1 (local linear).
#' @param bandwidth Positive scalar bandwidth.
#' @param kernel Kernel used for weighting. One of "triangular", "uniform",
#'               or "epanechnikov". Default is "triangular".
#' @param weights Optional non-negative observation weights. If NULL, all 1.
#'
#' @return A list with left/right coefficients and the estimated jump at 0.
#'
#' @export
local_polynomial_reg <- function(running_var,
                                 Y,
                                 order = 1,
                                 bandwidth,
                                 kernel = c("triangular", "uniform", "epanechnikov"),
                                 weights = NULL) {
  kernel <- match.arg(kernel)

  if (!is.numeric(running_var) || anyNA(running_var)) {
    stop("running_var must be a numeric vector with no missing values.")
  }
  if (!is.numeric(Y) || anyNA(Y) || length(Y) != length(running_var)) {
    stop("Y must be a numeric vector with no missing values and the same length as running_var.")
  }
  if (!is.numeric(order) || length(order) != 1 || order < 0 || order != floor(order)) {
    stop("order must be a non-negative integer.")
  }
  if (!is.numeric(bandwidth) || length(bandwidth) != 1 || bandwidth <= 0) {
    stop("bandwidth must be a positive numeric scalar.")
  }
  if (is.null(weights)) {
    weights <- rep(1, length(running_var))
  } else {
    if (!is.numeric(weights) || anyNA(weights) || length(weights) != length(running_var) || any(weights < 0)) {
      stop("weights must be a non-negative numeric vector with no missing values and the same length as running_var.")
    }
  }

  u <- abs(running_var) / bandwidth
  k <- switch(
    kernel,
    triangular = pmax(0, 1 - u),
    uniform = as.numeric(u <= 1),
    epanechnikov = 0.75 * pmax(0, 1 - u^2)
  )
  w <- weights * k

  poly_design <- function(r) {
    # Columns: 1, r, r^2, ..., r^order
    r <- as.numeric(r)
    out <- matrix(1, nrow = length(r), ncol = order + 1)
    if (order >= 1) {
      for (j in 1:order) {
        out[, j + 1] <- r^j
      }
    }
    out
  }

  fit_side <- function(side) {
    idx <- if (side == "left") running_var < 0 else running_var >= 0
    idx <- idx & (w > 0)
    n <- sum(idx)
    if (n == 0) {
      return(list(coef = rep(NA_real_, order + 1), n = 0))
    }
    x <- poly_design(running_var[idx])
    y <- Y[idx]
    ww <- w[idx]

    # Weighted least squares via lm.wfit.
    fit <- stats::lm.wfit(x = x, y = y, w = ww)
    list(coef = as.numeric(fit$coefficients), n = n)
  }

  left <- fit_side("left")
  right <- fit_side("right")

  mu_left <- left$coef[1]
  mu_right <- right$coef[1]

  list(
    order = order,
    bandwidth = bandwidth,
    kernel = kernel,
    n.left = left$n,
    n.right = right$n,
    coef.left = left$coef,
    coef.right = right$coef,
    mu.left = mu_left,
    mu.right = mu_right,
    tau = mu_right - mu_left
  )
}

