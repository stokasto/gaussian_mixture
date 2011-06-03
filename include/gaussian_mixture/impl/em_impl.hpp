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
    }

  template<int DIM>
    g_float
    EM<DIM>::Estep(const std::vector<typename GMM<DIM>::VectorType> &data)
    {
      g_float log_likeliehood = 0., likeliehood = 0.;
      int state = 0, iter = 0;
      for (; iter < (int) data.size(); ++iter)
        {
          // get the associated storage vector
          Eigen::VectorXd &storage_d = storage_[iter];
          // reset likeliehood accumulator
          // and calculate overall likeliehood of this data point
          likeliehood = 0.;
          for (state = 0; state < num_states_; ++state)
            {
              // query pdf of respective gaussian
              tmp_pdf_[state] = model_->getGaussian(state).pdf(data[iter]);
              // and accumulate likeliehood
              likeliehood += model_->getPrior(state) * tmp_pdf_[state];
            }
          log_likeliehood += log(likeliehood);

          // now that we now the overall likeliehood we can calculate
          // the weighted likeliehood that a given state produced the data point
          storage_d = model_->getPriors() * tmp_pdf_ / likeliehood;
        }
      return log_likeliehood;
    }

  template<int DIM>
    void
    EM<DIM>::Mstep(const std::vector<typename GMM<DIM>::VectorType> &data, bool &do_continue)
    {
      // TODO: implement this
    }

  template<int DIM>
    g_float
    EM<DIM>::runEM(const std::vector<typename GMM<DIM>::VectorType> &data, g_float epsilon,
        int max_iter)
    {
      if (!initialized_ || !model_ || num_states_ < 1 || max_iter < 1)
        return;

      int data_size = data.size();
      int iter = 0;
      g_float log_likeliehood = 0., old_log_likeliehood = 0., delta_log_likeliehood = 0.;
      bool do_continue = false;
      // allocate space in storage_ for em computation
      storage_.resize(data_size);
      for (int i = 0; i < data_size; ++i)
        {
          storage_[i].resize(num_states_);
        }

      while (iter < max_iter)
        {
          // asume we are done at first :)
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
