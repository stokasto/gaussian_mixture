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
#include <gaussian_mixture/gaussian_converter.h>

#include <vector>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace gmm;

template<int DIM>
  void
  gaussianTest()
  {
    int n = 10e5;
    std::vector<typename Gaussian<DIM>::VectorType> x(n);
    srand(time(0));
    // temporary N-dimensional "Vector" to store samples from the gaussian
    typename Gaussian<DIM>::VectorType sample;
    typename Gaussian<DIM>::VectorType mean;
    typename Gaussian<DIM>::MatrixType covariance;
    typename Gaussian<DIM>::MatrixType tmp;
    for (size_t k = 0; k < 5; ++k)
      {

        tmp = Gaussian<DIM>::MatrixType::Random();
        // TODO: the generated covariance matrix will be symmetric
        //       but its positive definiteness is not guaranteed, it is just very likely :)
        //       should we check this here using eigenvalues ?
        covariance = tmp.transpose() * tmp;
        for (int i = 0; i < sample.size(); ++i)
          mean(i) = random_uniform_0_k(40);
        // declare 1 dimensional gaussian
        Gaussian<DIM> gauss = Gaussian<DIM> ().setMean(mean).setCovariance(covariance);
        typename Gaussian<DIM>::VectorType sampleMean(Gaussian<DIM>::VectorType::Zero());
        for (int i = 0; i < n; ++i)
          {
            // draw from gaussian
            gauss.draw(sample);
            // store in vector
            x[i] = sample;
            // and add to mean
            sampleMean += sample;
          }
        // calculate correct mean
        sampleMean /= n;
        // calculate sample variance
        typename Gaussian<DIM>::MatrixType var(Gaussian<DIM>::MatrixType::Zero());
        for (int i = 0; i < n; ++i)
          {
            var += (x[i] - sampleMean) * (x[i] - sampleMean).transpose();
          }
        var /= n;

        // assertions
        for (int i = 0; i < mean.size(); ++i)
          {
            EXPECT_NEAR(mean(i), sampleMean(i), 10e-2);
          }
        for (int y = 0; y < covariance.rows(); ++y)
          {
            for (int x = 0; x < covariance.cols(); ++x)
              {
                EXPECT_NEAR(covariance(y,x), var(y,x), 10e-2);
              }
          }
      }
  }

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

      EXPECT_NEAR(0.0, mean, 10e-3);
      EXPECT_NEAR(1.0, var, 10e-3);
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
        {
          var += (x(i) - sampleMean) * (x(i) - sampleMean);
        }
      var = (1. / (n - 1)) * var;

      EXPECT_NEAR(mean(0), sampleMean, 10e-2);
      EXPECT_NEAR(covariance(0), var, 10e-2);
    }
}

TEST(Gaussian, gaussian2D)
{
  gaussianTest<2> ();
}

TEST(Gaussian, gaussianND)
{
  // TODO: these tests can actually take quite a while, is there a way in gtest
  //       to mark them as slow ?
  gaussianTest<3> ();
  gaussianTest<4> ();
  gaussianTest<5> ();
  gaussianTest<6> ();
}

TEST(Gaussian, gaussianProject)
{
  Gaussian<3>::VectorType mean;
  Gaussian<3>::MatrixType tmp;
  Gaussian<3>::MatrixType covariance;
  tmp = Gaussian<3>::MatrixType::Random();
  covariance = tmp.transpose() * tmp;
  mean = Gaussian<3>::VectorType::Random();

  Gaussian<3> gauss = Gaussian<3> ().setMean(mean).setCovariance(covariance);
  // project to first two dimensions
  Gaussian<2> projection;
  gauss.getConverter<2> ().project(projection);
  Gaussian<2>::MatrixType checkCovar = projection.getCovariance();
  Gaussian<2>::VectorType checkMean = projection.mean();
  for (int i = 0; i < checkMean.size(); ++i)
    {
      EXPECT_NEAR(checkMean(i), mean(i), 1e-3);
    }
  for (int i = 0; i < checkCovar.rows(); ++i )
    {
      for (int j = 0; j < checkCovar.cols(); ++j )
        {
          EXPECT_NEAR(checkCovar(i,j), covariance(i,j), 1e-3);
        }
    }

}

TEST(Gaussian, testPDF)
{
  Gaussian<1> gauss1;
  Gaussian<1>::VectorType tmp(Gaussian<1>::VectorType::Zero());
  Gaussian<2> gauss2;
  Gaussian<2>::VectorType tmp2(Gaussian<2>::VectorType::Zero());
  EXPECT_NEAR(0.4, gauss1.pdf(tmp), 10e-3);
  EXPECT_NEAR(0.15, gauss2.pdf(tmp2), 10e-3);
}

int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
