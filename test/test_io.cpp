#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gaussian_converter.h>
#include <gaussian_mixture/gmm.h>

#include <stdio.h>
#include <vector>
#include <string>
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

template<int DIM>
  Gaussian<DIM>
  getRandGaussian()
  {
    typename Gaussian<DIM>::MatrixType tmp = Gaussian<DIM>::MatrixType::Random();
    typename Gaussian<DIM>::VectorType mean = Gaussian<DIM>::VectorType::Random();
    typename Gaussian<DIM>::MatrixType var = tmp.transpose() * tmp;
    return Gaussian<DIM> ().setMean(mean).setCovariance(var);
  }

template<int DIM>
  GMM<DIM>
  getRandGMM()
  {
    Gaussian<3>::MatrixType tmp = Gaussian<3>::MatrixType::Random();
    Gaussian<3>::VectorType mean1 = Gaussian<3>::VectorType::Random();
    Gaussian<3>::MatrixType var1 = tmp.transpose() * tmp;
    Gaussian<3> gauss = Gaussian<3> ().setMean(mean1).setCovariance(var1);

    tmp = Gaussian<3>::MatrixType::Random();
    Gaussian<3>::VectorType mean2 = Gaussian<3>::VectorType::Random();
    Gaussian<3>::MatrixType var2 = tmp.transpose() * tmp;
    Gaussian<3> gauss2 = Gaussian<3> ().setMean(mean2).setCovariance(var2);

    GMM<3> gmm = GMM<3> ().setNumStates(2);
    gmm.setMean(0, mean1).setCovariance(0, var1);
    gmm.setMean(1, mean2).setCovariance(1, var2);
    return gmm;
  }

std::string
getTmpName()
{
  char buffer[L_tmpnam];
  tmpnam(buffer);

  // create a proper string from the temporary name
  return std::string(buffer);
}

#ifdef GMM_ROS
TEST(Gaussian, toMessage)
  {
    Gaussian<3> gauss = getRandGaussian<3>();

    gaussian_mixture::GaussianModel msg;
    // write to message
    EXPECT_TRUE(gauss.toMessage(msg));

    Gaussian<3> gauss2;
    // read from message again
    EXPECT_TRUE(gauss2.fromMessage(msg));

    EXPECT_EQ(msg.dim, 3);
    gaussEq(gauss, gauss2);
  }

TEST(GMM, toMessage)
  {
    GMM<3> gmm = getRandGMM<3>();

    gaussian_mixture::GaussianMixtureModel msg;
    // write to message
    EXPECT_TRUE(gmm.toMessage(msg));
    EXPECT_FALSE(msg.initialized);
    msg.initialized = true;
    gmm.forceInitialize();
    // and read again
    GMM<3> gmm2;
    EXPECT_TRUE(gmm2.fromMessage(msg));

    EXPECT_EQ(msg.dim, 3);
    EXPECT_EQ(msg.num_states, 2);
    EXPECT_EQ(gmm.getNumStates(), gmm2.getNumStates());

    for(int i = 0; i < 2; ++i)
      {
        EXPECT_EQ(gmm.getPrior(i), gmm2.getPrior(i));
        gaussEq(gmm.getGaussian(i), gmm2.getGaussian(i));
      }
  }

TEST(Gaussian, toBagFile)
  {
    std::string fname(getTmpName());

    Gaussian<3> gauss = getRandGaussian<3> ();

    // write to message
    ASSERT_TRUE(gauss.toBag(fname));

    Gaussian<3> gauss2;
    // read from message again
    ASSERT_TRUE(gauss2.fromBag(fname));

    gaussEq(gauss, gauss2);
  }

TEST(GMM, toBagFile)
  {
    std::string fname(getTmpName());

    GMM<3> gmm = getRandGMM<3>();

    gmm.forceInitialize();

    // write to file
    ASSERT_TRUE(gmm.toBag(fname));
    // and read again
    GMM<3> gmm2;
    ASSERT_TRUE(gmm2.fromBag(fname));

    EXPECT_EQ(gmm.getNumStates(), gmm2.getNumStates());
    for(int i = 0; i < 2; ++i)
      {
        EXPECT_EQ(gmm.getPrior(i), gmm2.getPrior(i));
        gaussEq(gmm.getGaussian(i), gmm2.getGaussian(i));
      }
  }
#endif

TEST(Gaussian, toBinFile)
{
  std::string fname(getTmpName());

  Gaussian<3> gauss = getRandGaussian<3> ();

  // write to message
  ASSERT_TRUE(gauss.toBinaryFile(fname));

  Gaussian<3> gauss2;
  // read from message again
  ASSERT_TRUE(gauss2.fromBinaryFile(fname));

  gaussEq(gauss, gauss2);
}

TEST(GMM, toBinFile)
{
  std::string fname(getTmpName());

  GMM<3> gmm = getRandGMM<3> ();

  gmm.forceInitialize();

  // write to file
  ASSERT_TRUE(gmm.toBinaryFile(fname));
  // and read again
  GMM<3> gmm2;
  ASSERT_TRUE(gmm2.fromBinaryFile(fname));

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
  srand(time(0));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
