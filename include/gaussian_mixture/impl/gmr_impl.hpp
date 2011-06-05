#ifndef GMR_IMPL_HPP_
#define GMR_IMPL_HPP_

#include <gaussian_mixture/gmr.h>

namespace gmm
{

  template<int DIM, int P_DIM>
    GMR<DIM, P_DIM>::GMR() :
      model_(0), initialized_(false)
    {
    }
  template<int DIM, int P_DIM>
    GMR<DIM, P_DIM>::~GMR()
    {
    }

  template<int DIM, int P_DIM>
    GMR<DIM, P_DIM> &
    GMR<DIM, P_DIM>::setInputGMM(const GMM<DIM> &model)
    {
      model_ = &model;

      // allocate gaussians
      num_states_ = model_->getNumStates();
      condGaussians_.resize(num_states_);
      marginalGaussians_.resize(num_states_);

      for (int i = 0; i < num_states_; ++i)
        {
          const Gaussian<DIM> &gauss = model_->getGaussian(i);
          converter_.push_back(gauss.template getConverter<P_DIM>());
          // precompute marginal
          converter_.back().getMarginalDistribution(marginalGaussians_[i]);
        }
      // resize weight vector
      weights_.resize(num_states_);
      // mark as initialized
      initialized_ = true;
      return *this;
    }

  template<int DIM, int P_DIM>
    void
    GMR<DIM, P_DIM>::query(const typename Gaussian<DIM-P_DIM>::VectorType &input,
        typename Gaussian<P_DIM>::VectorType &sample)
    {
      if (!initialized_ || !model_)
        return;
      if (num_states_ < 1)
        return;
      g_float partition = 0.;
      g_float accumulator = 0.;
      g_float thresh = random_uniform_0_1();
      int state = 0;

      // resize weight vector
      if (weights_.size() != num_states_)
        weights_.resize(num_states_);

      // first compute weights of all distributions
      for (state = 0; state < num_states_; ++state)
        {
          weights_(state) = marginalGaussians_[state].pdf(input);
          partition += weights_(state);
        }

      // next chose gaussian to sample from
      state = 0;
      while (accumulator < thresh)
        {
          accumulator += weights_(state) / partition;
          ++state;
        }
      // use last gaussian with which thresh was overcome
      --state;
      // get the correct conditional distribution
      converter_[state].getConditionalDistribution(input, condGaussians_[state]);

      // and draw a sample
      condGaussians_[state].draw(sample);
    }
  template<int DIM, int P_DIM>
    void
    GMR<DIM, P_DIM>::getConditionalDistribution(const typename Gaussian<DIM - P_DIM>::VectorType &input,
        Gaussian<P_DIM> &result)
    {
      g_float partition = 0.;
      int state;
      typename Gaussian<P_DIM>::VectorType mean;
      typename Gaussian<P_DIM>::MatrixType covariance;

      // zero out mean and covariance first
      mean.setZero();
      covariance.setZero();

      // get weights of all distributions
      for (state = 0; state < num_states_; ++state)
        {
          weights_(state) = marginalGaussians_[state].pdf(input);
          partition += weights_(state);
          // also update the conditional
          converter_[state].getConditionalDistribution(input, condGaussians_[state]);
          // and update the mean
          mean += weights_(state) * condGaussians_[state].getMean();
        }
      // scale mean
      mean /= partition;

      // and calculate variance
      for (state = 0; state < num_states_; ++state)
        {
          const typename Gaussian<P_DIM>::MatrixType &tmpCovar = condGaussians_[state].getCovariance();
          for (int i = 0; i < covariance.rows(); ++i)
            for (int j = 0; j < covariance.cols(); ++j)
              {
                g_float weight_sqr = (weights_(state) / partition);
                weight_sqr = weight_sqr * weight_sqr;
                covariance(i, j) += weight_sqr * tmpCovar(i, j);
              }
        }

      result.setMean(mean).setCovariance(covariance);
    }

  template<int DIM, int P_DIM>
    const GMM<DIM> &
    GMR<DIM, P_DIM>::getInputGMM() const
    {
      assert(initialized_ && model_ && "ERROR: you called getInputGMM() on an uninitialized GMR");
      return *model_;
    }

}

#endif /* GMR_IMPL_HPP_ */
