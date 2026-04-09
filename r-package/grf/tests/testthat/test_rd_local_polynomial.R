test_that("C++ local polynomial RD matches R WLS reference", {
  set.seed(123)
  n <- 2000
  running <- rnorm(n)
  y <- 1 + 0.3 * running + 2 * (running >= 0) + rnorm(n)

  order <- 1
  bandwidth <- 0.7
  kernel <- "triangular"

  lp_tau_r <- function(running, y, order, bandwidth, kernel) {
    u <- abs(running) / bandwidth
    k <- switch(
      kernel,
      triangular = pmax(0, 1 - u),
      uniform = as.numeric(u <= 1),
      epanechnikov = 0.75 * pmax(0, 1 - u^2)
    )

    design <- function(r) {
      x <- matrix(1, nrow = length(r), ncol = order + 1)
      if (order >= 1) {
        for (j in 1:order) x[, j + 1] <- r^j
      }
      x
    }

    fit_side <- function(idx) {
      x <- design(running[idx])
      stats::lm.wfit(x = x, y = y[idx], w = k[idx])$coefficients[1]
    }

    mu_left <- fit_side(running < 0 & k > 0)
    mu_right <- fit_side(running >= 0 & k > 0)
    mu_right - mu_left
  }

  tau_r <- lp_tau_r(running, y, order, bandwidth, kernel)
  tau_cpp <- grf:::rd_local_polynomial_tau(running, y, order, bandwidth, kernel, NULL)

  expect_equal(unname(tau_cpp), as.numeric(tau_r), tolerance = 1e-10)
})

test_that("C++ local polynomial RD matches rdrobust conventional estimate (if installed)", {
  skip_if_not_installed("rdrobust")

  set.seed(123)
  n <- 4000
  running <- rnorm(n)
  y <- 1 + 0.3 * running + 2 * (running >= 0) + rnorm(n)

  order <- 1
  bandwidth <- 0.7
  kernel <- "triangular"

  rd <- rdrobust::rdrobust(y = y, x = running, c = 0, p = order, h = bandwidth, kernel = kernel)

  tau_rdrobust <- NULL
  if (!is.null(rd$coef)) {
    coef <- rd$coef
    if (!is.null(rownames(coef)) && "Conventional" %in% rownames(coef)) {
      tau_rdrobust <- as.numeric(coef["Conventional", 1])
    } else {
      tau_rdrobust <- as.numeric(coef[1, 1])
    }
  }
  if (is.null(tau_rdrobust) || length(tau_rdrobust) != 1 || is.na(tau_rdrobust)) {
    skip("rdrobust output format not recognized for extracting conventional estimate")
  }

  tau_cpp <- grf:::rd_local_polynomial_tau(running, y, order, bandwidth, kernel, NULL)

  expect_equal(unname(tau_cpp), as.numeric(tau_rdrobust), tolerance = 1e-8)
})
