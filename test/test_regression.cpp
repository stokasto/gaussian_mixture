#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gmm.h>
#include <gaussian_mixture/gmr.h>
#include <gaussian_mixture/em.h>

using namespace gmm;

int main(void)
{
  int n_data = 1e5;
  Gaussian<2>::VectorType tmp;
  std::vector<Gaussian<2>::VectorType> train_data;

  GMM<2> gmm = GMM<2>().setNumStates(2);

  // generate training data from a simple parbola
  for(int i = 0; i < n_data; ++i)
    {
      g_float sign = (random_uniform_0_1() > 0.5) ? -1 : 1;
      g_float posx = sign * random_uniform_0_1() * 20.;
      tmp(0) = posx;
      tmp(1) = posx*posx;
      train_data.push_back(tmp);
    }
  // init gaussian mixture model
  gmm.initKmeans(train_data);

  // get EM instance
  EM<2> em = gmm.getEM();
  // run em
  std::cout << "Running em algorithm" << std::endl;
  em.runEM(train_data, 0.1, 300);
  std::cout << "DONE running em algorithm" << std::endl;

  // get regression model
  std::cout << "Getting regression model" << std::endl;
  GMR<2,1> gmr = gmm.getRegressionModel<1>();
  // and draw samples from it

  return 0;
}
