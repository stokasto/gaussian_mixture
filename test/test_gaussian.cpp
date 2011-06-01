// Test normality function --> taken from Manuel Blums
// libgp - Gaussian Process library for Machine Learning
// Copyright (C) 2010 Universität Freiburg
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.


#include <gaussian_mixture/gaussian.h>

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(Gaussian, random_normal)
{
  int n = 10e5;
  Eigen::VectorXd x(n);
  srand(time(0));
  for (size_t k = 0; k < 10; ++k)
    {
      // draw from normal distribution
      for (int i = 0; i < n; ++i)
        x(i) = gmm::random_normal();//gmm::box_muller(0.,1.);
      // calculate mean
      double mean = x.mean();
      // calculate sample variance
      double var = 0.;
      for (int i = 0; i < n; ++i)
        var += (x(i) - mean) * (x(i) - mean);
      var = (1. / (n - 1)) * var;

      ASSERT_NEAR(0.0, mean, 10e-3);
      ASSERT_NEAR(1.0, var, 10e-3);
    }
}

int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
