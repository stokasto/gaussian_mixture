#ifndef GMR_H_
#define GMR_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gaussian_converter.h>
#include <gaussian_mixture/gmm.h>

#include <Eigen/Core>

namespace gmm
{

  /** Definition of a Gaussian Mixture Regression model converting a DIM dimensional Gaussian
   *  to a P_DIM dimensional regression model.
   *
   *  @author Jost Tobias Springenberg
   */
  template<int DIM, int P_DIM>
    class GMR
    {
    private:
      /** pointer to the input model */
      const GMM<DIM> *model_;
      /** converters for all gaussians */
      std::vector<GaussianConverter<DIM, P_DIM> > converter_;
      /** storage for marginal and conditional distributions extracted from the model */
      std::vector<Gaussian<DIM - P_DIM> > marginalGaussians_;
      std::vector<Gaussian<P_DIM> > condGaussians_;
      // for convenience --> should be == condGaussians_.size()
      int num_states_;
      bool initialized_;
      // temporary storage
      Eigen::VectorXd weights_;
      typename Gaussian<P_DIM>::VectorType resultMean_;
      typename Gaussian<P_DIM>::MatrixType resultCovar_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      /** Create a new gaussian mixture regression instance.
       *  NOTE: setInputGMM needs to be called before a call to any other method.
       *
       */
      GMR();
      virtual
      ~GMR();

      /** Set the gmm model that the gmr should operate on.
       *  NOTE: this will alter the preallocated converters and marginal distributions.
       *
       *  @param model the gmm that the regression operates on
       *  @return a reference to this
       */
      GMR<DIM, P_DIM> &
      setInputGMM(const GMM<DIM> &model);

      /** Draw a sample from the regression model.
       *  The resulting sample will be of size P_DIM.
       *
       *  @param input the input vector of size DIM-P_DIM
       *         for which a regression sample should be computed
       *  @param sample this will be set to the resulting sample
       */
      void
      query(const typename Gaussian<DIM - P_DIM>::VectorType &input,
          typename Gaussian<P_DIM>::VectorType &sample);

      /** Get the conditional distribution used for regression given an input vector.
       *
       * @param input the input vector of size DIM-P_DIM
       *        for which the conditional distribution should be computed
       * @oaram result this will be set to the resulting conditional distribution
       */
      void
      getConditionalDistribution(const typename Gaussian<DIM - P_DIM>::VectorType &input, Gaussian<
          P_DIM> &result);

      /** Get a const reference to the Gaussian Mixture Model which the regression operates on.
       *
       * @return reference to the gmm
       */
      const GMM<DIM> &
      getInputGMM() const;

    };
}

// implementation in gmr_impl.hpp
#include <gaussian_mixture/impl/gmr_impl.hpp>

#endif /* GMR_H_ */
