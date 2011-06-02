#ifndef DISTRIBUTIONS_H_
#define DISTRIBUTIONS_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/random.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>

namespace gmm
{

  /* definition of a n dimensional gaussian */
  template<int DIM>
    class Gaussian
    {
    public:
      typedef Eigen::Matrix<g_float, DIM, DIM> MatrixType;
      typedef Eigen::Matrix<g_float, DIM, 1> VectorType;
      const int dim;

    private:
      VectorType mean_;
      MatrixType covariance_;
      Eigen::LLT<MatrixType> cholesky_; // preallocate cholesky decomposition
      g_float partition_;

      /* these are just for temporary storage
       * we do keep them as member variables such that the pdf() function
       * is real time safe!
       */
      VectorType beta_;
      VectorType alpha_;
      VectorType tmp_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      /* constructors / destructors*/
      Gaussian();
      virtual
      ~Gaussian();

      // Methods for following the named parameters paradigm
      Gaussian<DIM> &
      setCovariance(const MatrixType &cov);
      Gaussian<DIM> &
      setMean(const VectorType &mean);

      // methods
      // draw a sample from the distribution
      void
      draw(VectorType &res) const;
      // calculate the pdf function at position x
      g_float
      pdf(const VectorType x);

      // getter
      VectorType &
      mean();

      const MatrixType &
      getCovariance() const;
      const VectorType &
      getMean() const;

      template<int P_DIM>
        GaussianConverter<DIM, P_DIM>
        getConverter() const;

      //TODO: toFile method
    };

}

// --> see gaussian_impl.hpp for implementation
#include <gaussian_mixture/impl/gaussian_impl.hpp>

#endif
