#ifndef GAUSSIAN_CONVERTER_IMPL_HPP_
#define GAUSSIAN_CONVERTER_IMPL_HPP_

#include <gaussian_mixture/gaussian_converter.h>

namespace gmm
{

  template<int DIM, int P_DIM>
    GaussianConverter<DIM, P_DIM>::GaussianConverter() :
      initialized_(false), input_(0)
    {
      assert(P_DIM < DIM && "ERROR: the dimension of the target gaussian must be < source");
    }

  template<int DIM, int P_DIM>
    GaussianConverter<DIM, P_DIM>::~GaussianConverter()
    {
    }

  template<int DIM, int P_DIM>
    GaussianConverter<DIM, P_DIM> &
    GaussianConverter<DIM, P_DIM>::setInputGaussian(const Gaussian<DIM> &gauss)
    {
      input_ = &gauss;
      initialized_ = true;
      // sub-matrices from this gaussians covariance matrix
      // --> we divide the matrix into 4 parts to get the contribution
      //     of conditional and marginal distribution
      sigmaI_ = gauss.getCovariance().block(0, 0, DIM - P_DIM, DIM - P_DIM);
      sigmaO_ = gauss.getCovariance().block(DIM - P_DIM, DIM - P_DIM, P_DIM, P_DIM);
      sigmaOI_ = gauss.getCovariance().block(DIM - P_DIM, 0, P_DIM, DIM - P_DIM);
      sigmaIO_ = gauss.getCovariance().block(0, DIM - P_DIM, DIM - P_DIM, P_DIM);

      // precalculate cholesky for submatrix
      llt_ = sigmaI_.llt();
      return *this;
    }

  template<int DIM, int P_DIM>
    Gaussian<DIM> &
    GaussianConverter<DIM, P_DIM>::getInputGaussian() const
    {
      assert(initialized_ && input_ && "ERROR: you called getInputGaussian() on an uninitialized converter!");
      return *input_;
    }

  template<int DIM, int P_DIM>
    void
    GaussianConverter<DIM, P_DIM>::project(const typename Gaussian<P_DIM>::VectorType &dims,
        Gaussian<P_DIM> &result)
    {
      if (!initialized_ || !input_)
        return;
      assert(P_DIM < DIM && "ERROR: projected dim is bigger than original dim");
      assert(dims.size() == P_DIM && "ERROR: projected dim does not match provided vector");

      const SourceMatType &covariance = input_->getCovariance();
      const SourceVecType &mean = input_->getMean();

      // copy covariance matrix
      for (int i = 0; i < dims.size(); ++i)
        for (int j = 0; j < dims.size(); ++j)
          {
            assert(dims(i) >= 0 && dims(i) < DIM);
            resultCovar_(i, j) = covariance(dims(i), dims(j));
          }
      result.setCovariance(resultCovar_);
      // copy means
      for (int i = 0; i < dims.size(); ++i)
        result.mean()(i) = mean(dims(i));
    }

  template<int DIM, int P_DIM>
    void
    GaussianConverter<DIM, P_DIM>::project(Gaussian<P_DIM> &result)
    {
      if (!initialized_ || !input_)
        return;
      for (int i = 0; i < P_DIM; ++i)
        tmp_out_(i) = i;
      // TODO: this could be done more efficiently by copying
      //       the covariance matrix block wise
      project(tmp_out_, result);
    }

  template<int DIM, int P_DIM>
    void
    GaussianConverter<DIM, P_DIM>::getConditionalDistribution(
        const typename Gaussian<DIM - P_DIM>::VectorType &input, Gaussian<P_DIM> &result)
    {
      if (!initialized_ || !input_)
        return;

      tmp_in_ = input - input_->getMean().head(DIM - P_DIM);
      // calculate mean
      beta_ = llt_.matrixL().solve(tmp_in_);
      resultMean_ = input_->getMean().tail(P_DIM) + sigmaOI_ * llt_.matrixL().transpose().solve(
          beta_);

      // calculate covariance
      tmp_llt_sigmaIO_ = llt_.matrixL().solve(sigmaIO_);
      resultCovar_ = sigmaO_ - sigmaOI_ * llt_.matrixL().transpose().solve(tmp_llt_sigmaIO_);
      /* // DEBUG ONLY
      std::cout << "mean of conditional: " << std::endl;
      std::cout << resultMean_ << std::endl;
      std::cout << "covar of conditional: " << std::endl;
      std::cout << resultCovar_ << std::endl;
      */
      result.setMean(resultMean_).setCovariance(resultCovar_);
    }

  template<int DIM, int P_DIM>
    void
    GaussianConverter<DIM, P_DIM>::getMarginalDistribution(Gaussian<DIM - P_DIM> &result)
    {
      const typename Gaussian<DIM>::VectorType &mean = input_->getMean();
      result.setMean(mean.head(DIM-P_DIM));
      result.setCovariance(sigmaI_);
    }
}

#endif
