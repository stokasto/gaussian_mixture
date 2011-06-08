#ifndef GAUSSIAN_CONVERTER_H_
#define GAUSSIAN_CONVERTER_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>

#include <Eigen/Core>

namespace gmm
{

  /** Definition of a Converter Class converting a DIM dimensional Gaussian
   *  to a P_DIM dimensional version.
   *
   *  @author Jost Tobias Springenberg
   */
  template<int DIM, int P_DIM>
    class GaussianConverter
    {
    private:
      // typedef all used vectors for convenience
      typedef typename Gaussian<DIM>::VectorType SourceVecType;
      typedef typename Gaussian<DIM>::MatrixType SourceMatType;
      typedef typename Gaussian<DIM - P_DIM>::VectorType MarginalVecType;
      typedef typename Gaussian<DIM - P_DIM>::MatrixType MarginalMatType;
      typedef typename Gaussian<P_DIM>::VectorType TargetVecType;
      typedef typename Gaussian<P_DIM>::MatrixType TargetMatType;
      bool initialized_;
      /** pointer to the input gaussian */
      const Gaussian<DIM> *input_;
      // pointers to submatrices of the input gaussian
      //Eigen::Block<Eigen::Matrix<g_float, DIM - P_DIM, DIM - P_DIM>, DIM - P_DIM, DIM - P_DIM, false, true>
      Eigen::Matrix<g_float, DIM - P_DIM, DIM - P_DIM> sigmaI_;
      Eigen::Matrix<g_float, P_DIM, P_DIM> sigmaO_;
      Eigen::Matrix<g_float, P_DIM, DIM - P_DIM> sigmaOI_;
      Eigen::Matrix<g_float, DIM - P_DIM, P_DIM> sigmaIO_;
      // preallocate temporary storage to make everything as fast as possible
      TargetVecType tmp_out_;
      TargetVecType resultMean_;
      TargetMatType resultCovar_;
      Eigen::Matrix<g_float, DIM - P_DIM, P_DIM> tmp_llt_sigmaIO_;
      Eigen::LLT<MarginalMatType> llt_;
      MarginalMatType beta_;
      MarginalMatType tmp_in_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      /** Create a new Converter instance. */
      GaussianConverter();
      virtual
      ~GaussianConverter();

      // for named variable pattern

      /** Set the input gaussian that this converter operates on.
       *  NOTE: This will also reset the referenes to the submatrices of the covariance matrix.
       *
       * @param gauss the gaussian that this converter should operate on
       * @return returns a reference to this
       */
      GaussianConverter<DIM, P_DIM> &
      setInputGaussian(const Gaussian<DIM> &gauss);

      /** Project a gaussian to a lower dimensional version.
       *
       * @param dims the dimensions that the lower dimensional version should contain
       * @param result this will contain the resulting gaussian after the call
       */
      void
      project(const typename Gaussian<P_DIM>::VectorType &dims, Gaussian<P_DIM> &result);

      /** Project to the first p dimensions.
       *
       * @param result this will contain the resulting gaussian after the call
       */
      void
      project(Gaussian<P_DIM> &result);

      /** Get the conditional distribution of the last P_DIM dimensions
       *  given the input vector of the first DIM-P_DIM dimensions.
       *
       * @param input the input vector that the gaussian should be conditioned on
       * @param result this will contain the computed conditional distribution after the call
       */
      void
      getConditionalDistribution(const typename Gaussian<DIM - P_DIM>::VectorType &input, Gaussian<
          P_DIM> &result);

      /** Compute the marginal distribution of the first DIM-P_DIM dimensions.
       *
       * @param result this will contain the computed marginal distribution after the call
       */
      void
      getMarginalDistribution(Gaussian<DIM - P_DIM> &result);

      // getter
      /** Get a reference to the input gaussian that this converter operates on.
       *
       * @return the input gaussian
       */
      Gaussian<DIM> &
      getInputGaussian() const;

    };
}

// include implementation
#include <gaussian_mixture/impl/gaussian_converter_impl.hpp>

#endif /* GAUSSION_CONVERTER_H_ */
