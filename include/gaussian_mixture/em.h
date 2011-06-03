#ifndef EM_H_
#define EM_H_

#include <gaussian_mixture/gmm.h>

#include <Eigen/Core>
#include <vector>

namespace gmm
{
  template<int DIM>
    class EM
    {
    private:
      // pointer to the gaussian mixture model that this em instance operates on
      GMM<DIM> *model_;

      // storage for em computations
      // TODO: this might get pretty big, how large is the overhead
      //       inflicted by usage of vector and Eigen::VectorXd ?
      //       If it is too large we might resort to just allocating one large float array here
      std::vector<Eigen::VectorXd> storage_;
      // another temporary vector that has size of num_states_
      Eigen::VectorXd tmp_pdf_;

      // the gmm size defined once again :) --> just for convenience
      int num_states_;
      bool initialized_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      EM();
      virtual
      ~EM();

      EM<DIM> &
      setInputGMM(GMM<DIM> &model);

      g_float
      Estep(const std::vector<typename GMM<DIM>::VectorType> &data);
      void
      Mstep(const std::vector<typename GMM<DIM>::VectorType> &data, bool &do_continue);

      g_float
      runEM(const std::vector<typename GMM<DIM>::VectorType> &data, g_float epsilon, int max_iter);

    };
}

// see em_impl.hpp for implementation
#include <gaussian_mixture/impl/em_impl.hpp>

#endif /* EM_H_ */
