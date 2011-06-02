#ifndef GAUSSIAN_IMPL_HPP_
#define GAUSSIAN_IMPL_HPP_

#include <gaussian_mixture/gaussian.h>

namespace gmm
{

  /* gaussian implementation */
  template<int DIM>
    Gaussian<DIM>::Gaussian() :
      dim(DIM), mean_(VectorType::Zero()), covariance_(MatrixType::Identity()), cholesky_(
          covariance_.llt()), partition_(1)
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
    Gaussian<DIM>::pdf(const Gaussian<DIM>::VectorType x)
    {
      // precompute distance to mean
      tmp_ = x - mean_;
      // next compute (x - mean) * sigma^-1 * (x - mean)
      // using the cholesky decomposition
      beta_ = cholesky_.matrixL().solve(tmp_);
      alpha_ = cholesky_.matrixL().transpose().solve(beta_);
      g_float res = tmp_ * alpha_.dot(tmp_);
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
    Gaussian<DIM>::getCovariance() const
    {
      return covariance_;
    }

  template<int DIM>
    const typename Gaussian<DIM>::VectorType &
    Gaussian<DIM>::getMean() const
    {
      return mean_;
    }

  template<int DIM>
    Gaussian<DIM> &
    Gaussian<DIM>::setCovariance(const typename Gaussian<DIM>::MatrixType &cov)
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
    Gaussian<DIM>::setMean(const typename Gaussian<DIM>::VectorType &mean)
    {
      mean_ = mean;
      return *this;
    }

  template<int DIM>
    template<int P_DIM>
      GaussianConverter<DIM, P_DIM>
      Gaussian<DIM>::getConverter() const
      {
        return GaussianConverter<DIM, P_DIM> ().setInputGaussian(*this);
      }
}

#endif /* GAUSSIAN_IMPL_HPP_ */
