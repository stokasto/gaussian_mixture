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

using namespace gmm;

TEST(Gaussian, random_normal)
{
  int n = 10e5;
  Eigen::VectorXd x(n);
  srand(time(0));
  for (size_t k = 0; k < 10; ++k)
    {
      // draw from normal distribution
      for (int i = 0; i < n; ++i)
        x(i) = random_normal();
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

TEST(Gaussian, gaussian1D)
{
  int n = 10e5;
  Eigen::VectorXd x(n);
  srand(time(0));
  // temporary 1d "Vector" to store samples from the gaussian
  Gaussian<1>::VectorType sample;
  Gaussian<1>::VectorType mean;
  Gaussian<1>::MatrixType covariance;
  // assert that sample is in fact 1 dimensional
  ASSERT_EQ(1, sample.size());
  for (size_t k = 0; k < 10; ++k)
    {
      covariance(0) = 1 + random_uniform_0_1();
      mean(0) = random_uniform_0_k(100);
      // declare 1 dimensional gaussian
      Gaussian<1> gauss = Gaussian<1>().setMean(mean).setCovariance(covariance);
      for (int i = 0; i < n; ++i)
        {
          // draw from gaussian
          gauss.draw(sample);
          // store in vector since sample is 1 d
          x(i) = sample(0);
        }
      // calculate mean
      double sampleMean = x.mean();
      // calculate sample variance
      double var = 0.;
      for (int i = 0; i < n; ++i)
        var += (x(i) - sampleMean) * (x(i) - sampleMean);
      var = (1. / (n - 1)) * var;

      ASSERT_NEAR(mean(0), sampleMean, 10e-2);
      ASSERT_NEAR(covariance(0), var, 10e-2);
    }
}


int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
