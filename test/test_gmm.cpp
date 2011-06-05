#include <gaussian_mixture/gmm.h>
#include <gaussian_mixture/gmr.h>
#include <gaussian_mixture/em.h>

#include <Eigen/Dense>
#include <gtest/gtest.h>

int
main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
