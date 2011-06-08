#ifndef EM_H_
#define EM_H_

#include <gaussian_mixture/gmm.h>

#include <Eigen/Core>
#include <vector>

namespace gmm
{

  /** An Expectaction Maximization implementation for Gaussian Mixture Models
   *
   *  @author Jost Tobias Springenberg
   */
  template<int DIM>
    class EM
    {
    private:
      /** pointer to the gaussian mixture model that this em instance operates on */
      GMM<DIM> *model_;

      // storage for em computations
      // TODO: this might get pretty big, how large is the overhead
      //       inflicted by usage of Eigen::MatrixXd ?
      //       If it is too large we might resort to just allocating one large float array here
      Eigen::MatrixXd storage_;
      // another temporary vector that has size of num_states_
      Eigen::VectorXd tmp_pdf_;

      // the gmm size defined once again :) --> just for convenience
      int num_states_;
      bool initialized_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      /** Create a new EM instance.
       *  NOTE: setInputGMM needs to be called before any other method after creation.
       */
      EM();
      virtual
      ~EM();

      /** Set the gmm model that the em algorithm should operate on.
       *
       *  @param model the gmm
       *  @return a reference to this
       */
      EM<DIM> &
      setInputGMM(GMM<DIM> &model);

      /** Execute the expectation step of the em algorihtm.
       *
       * @param data the dataset on which the em algorithm is run
       * @return returns the log likeliehood of the data
       */
      g_float
      Estep(const std::vector<typename Gaussian<DIM>::VectorType> &data);

      /** Execute the maximization step of the em algorihtm.
       *
       * @param data the dataset on which the em algorithm is run
       * @param do_continue will be set to true if the m step reinitialized some gaussians
       */
      void
      Mstep(const std::vector<typename Gaussian<DIM>::VectorType> &data, bool &do_continue);

      /** Run the em algorith on the supplied data set.
       *
       * @param data the dataset that the em algorithm will use
       * @param epsilon the desired log likeliehood difference between iterations
       *        (whenthis epsilon reached the em computation will stop)
       * @param max_iter the maximum number of em steps
       * @return returns the log likeliehood of the training data
       */
      g_float
      runEM(const std::vector<typename Gaussian<DIM>::VectorType> &data, g_float epsilon,
          int max_iter);

    };
}

// see em_impl.hpp for implementation
#include <gaussian_mixture/impl/em_impl.hpp>

#endif /* EM_H_ */
