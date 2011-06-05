#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gmm.h>
#include <gaussian_mixture/gmr.h>
#include <gaussian_mixture/em.h>

#include <fstream>

using namespace gmm;

int main(void)
{
  int n_data = 1e5;
  Gaussian<2>::VectorType tmp;
  std::vector<Gaussian<2>::VectorType> train_data;
  std::ofstream data;
  std::ofstream regression;
  data.open("data.csv"); 
  regression.open("regression.csv"); 
  srand(time(0));

  GMM<2> gmm = GMM<2>().setNumStates(4);

  // generate training data from a simple parbola
  for(int i = 0; i < n_data; ++i)
    {
      g_float sign = (random_uniform_0_1() > 0.5) ? -1 : 1;
      g_float posx = sign * random_uniform_0_1() * 4.;
      tmp(0) = posx;
      tmp(1) = exp(posx) + random_uniform_0_1();
      train_data.push_back(tmp);
      data << tmp(0) << "\t" << tmp(1) << std::endl;
    }
  data.flush();
  data.close();
  // init gaussian mixture model
  gmm.initKmeans(train_data);

  // get EM instance
  EM<2> em = gmm.getEM();
  // run em
  std::cout << "Running em algorithm" << std::endl;
  em.runEM(train_data, 0.01, 300);
  std::cout << "DONE running em algorithm" << std::endl;
  // get regression model
  std::cout << "Getting regression model" << std::endl;
  GMR<2,1> gmr = gmm.getRegressionModel<1>();
  Gaussian<1>::VectorType source;
  Gaussian<1>::VectorType target;
  Gaussian<2>::VectorType sample;
  Gaussian<1> cond;
  // and draw samples from it
  for(int i = 0; i < n_data; ++i)
    {
      g_float sign = (random_uniform_0_1() > 0.5) ? -1 : 1;
      g_float posx = sign * random_uniform_0_1() * 7.;
      source(0) = posx;
      gmm.draw(sample);
      //gmr.getConditionalDistribution(source, cond);
      //cond.draw(target);
      regression << sample(0) << "\t" << sample(1) << std::endl;
      //regression << source(0) << "\t" << target(0) << std::endl;
    }
  regression.flush();
  regression.close();

  return 0;
}
