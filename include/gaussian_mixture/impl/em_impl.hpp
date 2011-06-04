#ifndef EM_IMPL_HPP_
#define EM_IMPL_HPP_

#include <gaussian_mixture/em.h>

namespace gmm
{

  template<int DIM>
    EM<DIM>::EM() :
      model_(0), initialized_(false)
    {
    }

  template<int DIM>
    EM<DIM>::~EM()
    {
    }

  template<int DIM>
    EM<DIM> &
    EM<DIM>::setInputGMM(GMM<DIM> &model)
    {
      model_ = &model;
      initialized_ = true;
      num_states_ = model.getNumStates();
      tmp_pdf_.resize(num_states_);
      return *this;
    }

  template<int DIM>
    g_float
    EM<DIM>::Estep(const std::vector<typename Gaussian<DIM>::VectorType> &data)
    {
      g_float log_likeliehood = 0., likeliehood = 0.;
      int state, iter;
      for (iter = 0; iter < (int) data.size(); ++iter)
        {
          // reset likeliehood accumulator
          // and calculate overall likeliehood of this data point
          likeliehood = 0.;
          for (state = 0; state < num_states_; ++state)
            {
              // query pdf of respective gaussian
              tmp_pdf_[state] = model_->gaussian(state).pdf(data[iter]);
              // and accumulate likeliehood
              likeliehood += model_->getPrior(state) * tmp_pdf_[state];
            }
          log_likeliehood += log(likeliehood);

          // now that we now the overall likeliehood we can calculate
          // the weighted likeliehood that a given state produced the data point
          storage_.col(iter) = model_->getPriors() * tmp_pdf_ / likeliehood;
        }
      return log_likeliehood;
    }

  template<int DIM>
    void
    EM<DIM>::Mstep(const std::vector<typename Gaussian<DIM>::VectorType> &data, bool &do_continue)
    {
      // TODO: implement this
      int rand_pos = 0, state, iter;
      typename Gaussian<DIM>::VectorType tmp;
      typename Gaussian<DIM>::VectorType mean;
      typename Gaussian<DIM>::MatrixType covariance;
      for (state = 0; state < num_states_; ++state)
        {
          g_float likeliehood = 0;
          // reset prior
          model_->setPrior(state, 0.);
          // reset mean
          mean.setZero();
          // reset covariance
          covariance.setZero();
          // calculate new covariance matrix and mean
          // first calculate the new mean maximizing the expectation
          for (iter = 0; iter < (int) data.size(); ++iter)
            {
              // --> mean as weighted sum
              mean += storage_(state, iter) * data[iter];
              likeliehood += storage_(state, iter);
            }
          // normalize mean
          mean /= likeliehood;
          // next calculate covariance maximizing the expectation
          for (iter = 0; iter < (int) data.size(); ++iter)
            {
              tmp = data[iter] - mean;
              covariance += storage_(state, iter) * (tmp * tmp.transpose());
            }
          // normalize covariance
          covariance /= likeliehood;
          // set new mean and covariance
          model_->gaussian(state).setMean(mean).setCovariance(covariance);
          // set the prior to be the overall likeliehood
          model_->setPrior(state, likeliehood / data.size());

          // check if likeliehood is 0, which means this gaussian is not generating
          // any of the data points
          // --> if so reset the mean to a random data point and require one more em step
          if (likeliehood <= 0.)
            {
              rand_pos = rand() % data.size();
              model_->gaussian(state).setMean(data[rand_pos]);
              do_continue = true;
            }
        }
    }

  template<int DIM>
    g_float
    EM<DIM>::runEM(const std::vector<typename Gaussian<DIM>::VectorType> &data, g_float epsilon,
        int max_iter)
    {
      if (!initialized_ || !model_ || num_states_ < 1 || max_iter < 1)
        return 0.;

      int data_size = data.size();
      int iter = 0;
      g_float log_likeliehood = 0., old_log_likeliehood = 0., delta_log_likeliehood = 0.;
      bool do_continue = false;
      // allocate space in storage_ for em computation
      storage_.resize(num_states_, data_size);

      while (iter < max_iter)
        {
          // assume we are done at first :)
          do_continue = false;

          // execute expectation step
          log_likeliehood = Estep(data);
          delta_log_likeliehood = log_likeliehood - old_log_likeliehood;

          // convergence criterion
          // delta < epsilon and m step did not request continuation
          if (fabs(delta_log_likeliehood) < epsilon && !do_continue)
            break; // CONVERGENCE REACHED

          // execute maximization step
          Mstep(data, do_continue);

          // remember last likeliehood
          old_log_likeliehood = log_likeliehood;
          // next iteration
          ++iter;
        }
      // return final log likeliehood of the data
      return log_likeliehood;
    }

}

#endif /* EM_IMPL_HPP_ */
