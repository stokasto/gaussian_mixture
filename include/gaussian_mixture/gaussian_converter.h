#ifndef GAUSSIAN_CONVERTER_H_
#define GAUSSIAN_CONVERTER_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>

#include <Eigen/Core>

namespace gmm
{

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
      // pointer to the input gaussian
      bool initialized_;
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

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      GaussianConverter();
      virtual
      ~GaussianConverter();

      // for named variable pattern
      GaussianConverter<DIM, P_DIM> &
      setInputGaussian(const Gaussian<DIM> &gauss);

      // project a gaussian to a lower dimensional version
      void
      project(const typename Gaussian<P_DIM>::VectorType &dims, Gaussian<P_DIM> &result);
      // project to the first p dimensions
      void
      project(Gaussian<P_DIM> &result);
      // get the conditional distribution
      void
      getConditionalDistribution(const typename Gaussian<DIM - P_DIM>::VectorType &input,
          Gaussian<P_DIM> &result);
      void
      getMarginalDistribution(Gaussian<DIM-P_DIM> &result);

      // getter
      Gaussian<DIM> &
      getInputGaussian() const;

    };
}

// include implementation
#include <gaussian_mixture/impl/gaussian_converter_impl.hpp>

#endif /* GAUSSION_CONVERTER_H_ */
