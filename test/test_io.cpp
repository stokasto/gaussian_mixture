#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gaussian_converter.h>
#include <gaussian_mixture/gmm.h>
#include <gaussian_mixture/GaussianModel.h>

#include <vector>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using namespace gmm;

template<int DIM>
  void
  gaussEq(const Gaussian<DIM> &g1, const Gaussian<DIM> &g2)
  {
    for (int i = 0; i < DIM; ++i)
      {
        EXPECT_EQ(g1.getMean()(i), g2.getMean()(i));
      }
    for (int i = 0; i < DIM; ++i )
      {
        for (int j = 0; j < DIM; ++j )
          {
            EXPECT_EQ(g2.getCovariance()(i,j), g2.getCovariance()(i,j));
          }
      }
  }

TEST(Gaussian, toMessage)
{
  Gaussian<3>::MatrixType tmp = Gaussian<3>::MatrixType::Random();
  Gaussian<3>::VectorType mean = Gaussian<3>::VectorType::Random();
  Gaussian<3>::MatrixType var = tmp.transpose() * tmp;
  Gaussian<3> gauss = Gaussian<3> ().setMean(mean).setCovariance(var);

  gaussian_mixture::GaussianModel msg;
  // write to message
  gauss.toMessage(msg);

  Gaussian<3> gauss2;
  // read from message again
  gauss2.fromMessage(msg);

  EXPECT_EQ(msg.dim, 3);
  gaussEq(gauss, gauss2);
}

TEST(GMM, toMessage)
{
  Gaussian<3>::MatrixType tmp = Gaussian<3>::MatrixType::Random();
  Gaussian<3>::VectorType mean1 = Gaussian<3>::VectorType::Random();
  Gaussian<3>::MatrixType var1 = tmp.transpose() * tmp;
  Gaussian<3> gauss = Gaussian<3> ().setMean(mean1).setCovariance(var1);

  tmp = Gaussian<3>::MatrixType::Random();
  Gaussian<3>::VectorType mean2 = Gaussian<3>::VectorType::Random();
  Gaussian<3>::MatrixType var2 = tmp.transpose() * tmp;
  Gaussian<3> gauss2 = Gaussian<3> ().setMean(mean2).setCovariance(var2);

  GMM<3> gmm = GMM<3>().setNumStates(2);
  gmm.setMean(0,mean1).setCovariance(0, var1);
  gmm.setMean(1,mean2).setCovariance(1, var2);

  gaussian_mixture::GaussianMixtureModel msg;
  // write to message
  gmm.toMessage(msg);
  EXPECT_FALSE(msg.initialized);
  msg.initialized = true;
  gmm.forceInitialize();
  // and read again
  GMM<3> gmm2;
  gmm2.fromMessage(msg);

  EXPECT_EQ(msg.dim, 3);
  EXPECT_EQ(msg.num_states, 2);
  EXPECT_EQ(gmm.getNumStates(), gmm2.getNumStates());

  for(int i = 0; i < 2; ++i)
    {
      EXPECT_EQ(gmm.getPrior(i), gmm2.getPrior(i));
      gaussEq(gmm.getGaussian(i), gmm2.getGaussian(i));
    }
}

int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
