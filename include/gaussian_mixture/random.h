#ifndef GMM_RANDOM_H_
#define GMM_RANDOM_H_

#include <math.h>
#include <stdlib.h>

namespace gmm
{

  g_float
  random_uniform_0_1()
  {
    return g_float(rand()) / g_float(RAND_MAX);
  }

  g_float
  random_uniform_0_k(g_float k)
  {
    return random_uniform_0_1() * k;
  }

  g_float
  random_uniform_mk_k(g_float k)
  {
    g_float sign = 1.f;
    if (random_uniform_0_1() > 0.5f)
      sign = -1.f;
    return random_uniform_0_k(k) * sign;
  }

  // this function computes a random number taken from a normal distribution
  // using the Box-Mueller method. NOTE: a rejection method is used here,
  // as it is much faster than claculating sin and cos
  g_float
  random_normal()
  {
    g_float u1, u2, v1, v2;
    g_float r = 2.f;
    // get
    while (r >= 1.f || r == 0.f)
      { // reject v1 and v2 that do not suffice r = v1^2 + v2^2 <= 1
        // first get 2 uniform random vars
        u1 = random_uniform_0_1();
        u2 = random_uniform_0_1();
        // transform them to the interval [-1,1]
        v1 = 2.0f * u1 - 1.f;
        v2 = 2.0f * u2 - 1.f;
        // calculate r = v1^2 + v2^2
        r = v1 * v1 + v2 * v2;
      }
    return v1 * sqrt((-2.0f * log(r)) / r);
  }

  g_float
  random_normal(g_float mu, g_float sigma)
  {
    g_float z1 = random_normal();
    return mu + sigma * z1;
  }

}
#endif /* GMM_RANDOM_H_ */
