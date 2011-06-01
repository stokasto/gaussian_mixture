#ifndef GAUSSIAN_IMPL_HPP_
#define GAUSSIAN_IMPL_HPP_

#include <gaussian_mixture/gaussian.h>

namespace gmm
{
  /* helper implementation */

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

  /* gaussian implementation */
  template<int DIM>
    Gaussian<DIM>::Gaussian() :
      dim(DIM), mean_(MatrixType::Zero()), covariance_(MatrixType::Identity()),
          cholesky_(covariance_.llt()), partition_(1)
    {
    }

  template<int DIM>
    Gaussian<DIM>::~Gaussian()
    {
    }

  template<int DIM>
    void 
    Gaussian<DIM>::draw(typename Gaussian<DIM>::VectorType &res) const
    {
      // at first draw N random variables from a normal distribution
      for (int i = 0; i < dim; ++i)
        res(i) = random_normal();
      // multiply with covariance matrix and add mean
      // to get a proper sample from the current distribution
      res = cholesky_.matrixL() * res + mean_;
    }

  template<int DIM>
    g_float
    Gaussian<DIM>::pdf(Gaussian<DIM>::VectorType x) const
    {
      // precompute distance to mean
      VectorType tmp = x - mean_;
      // next compute (x - mean) * sigma^-1 * (x - mean)
      // using the cholesky decomposition
      VectorType beta = cholesky_.matrixL().llt().solve(x);
      VectorType alpha = cholesky_.matrixL().transpose().llt().solve(beta);
      g_float res = alpha.dot(x);
      // finally calculate pdf response
      res *= 0.5;
      res = exp(-res) / partition_;
      if (res == 0.)
        res = -1e7;
      return res;
    }

  template<int DIM>
    typename Gaussian<DIM>::VectorType &
    Gaussian<DIM>::mean()
    {
      return mean_;
    }

  template<int DIM>
    const typename Gaussian<DIM>::MatrixType &
    Gaussian<DIM>::getCovariance()
    {
      return covariance_;
    }

  template<int DIM>
    Gaussian<DIM> &
    Gaussian<DIM>::setCovariance(typename Gaussian<DIM>::MatrixType &cov)
    {
      g_float tmp = 1.;
      covariance_ = cov;
      // precompute cholesky
      cholesky_.compute(covariance_);
      // TODO: assert that cov is actually symmetric positive definite
      // recompute partition
      for (int i = 0; i < dim; ++i)
        { // first compute determinant in tmp
          tmp *= cov(i, i);
        }
      // square --> variance
      tmp = tmp * tmp;
      partition_ = sqrt(pow(M_PI, dim) * tmp);
      return *this;
    }

  template<int DIM>
    Gaussian<DIM> &
    Gaussian<DIM>::setMean(typename Gaussian<DIM>::VectorType &mean)
    {
      mean_ = mean;
      return *this;
    }

  template<int DIM>
    template<int P_DIM>
      Gaussian<P_DIM>
      Gaussian<DIM>::project(Eigen::VectorXd &dims) const
      {
        assert (P_DIM < DIM && "ERROR: projected dim is bigger than original dim");
        assert (dims.size() == P_DIM && "ERROR: projected dim does not match provided vector");
        // allocate resulting gaussian
        Gaussian<P_DIM> result;
        typename Gaussian<P_DIM>::MatrixType new_cov;
        // copy covariance matrix
        for (int i = 0; i < dims.size(); ++i)
          for (int j = 0; j < dims.size(); ++j)
            {
              assert(dims(i) >= 0 && dims(i) < DIM);
              new_cov(i, j) = covariance_(dims(i), dims(j));
            }
        result.setCovariance(new_cov);
        // copy means
        for (int i = 0; i < dims.size(); ++i)
          result.mean()(i) = mean_(dims(i));
        return result;
      }

  template<int DIM>
    template<int P_DIM>
      Gaussian<P_DIM>
      Gaussian<DIM>::project() const
      {
        Eigen::VectorXd dims(P_DIM);
        for (int i = 0; i < P_DIM; ++i)
          dims(i) = i;
        // TODO: this could be done more efficiently by copying
        //       the covariance matrix block wise
        return project<P_DIM> (dims);
      }
}

#endif /* GAUSSIAN_IMPL_HPP_ */
