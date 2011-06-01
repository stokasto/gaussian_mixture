#ifndef DISTRIBUTIONS_H_
#define DISTRIBUTIONS_H_

#include <cmath>
#include <Eigen/Core>

namespace gmm
{
  typedef float g_float;

  /* basic helper functions to draw 1-D random numbers */
  // uniform distribution
  g_float
  random_uniform_0_1();
  g_float
  random_uniform_0_k(double k);
  g_float
  random_uniform_mk_k(double k);

  // normal distribution
  g_float
  random_normal();
  g_float
  random_normal(g_float mu, g_float sigma);

  /* definition of a n dimensional gaussian */
  template<int DIM>
    class Gaussian
    {
    public:
      typedef Eigen::Matrix<g_float, DIM, DIM> MatrixType;
      typedef Eigen::Vector<g_float, DIM> VectorType;
      const int dim;

    private:
      VectorType mean_;
      MatrixType covariance_;
      MatrixType::LLT cholesky_; // preallocat cholesky decomposition
      g_float partition_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      /* constructors / destructors*/
      Gaussian();
      virtual
      ~Gaussian();

      // Methods for following the named parameters paradigm
      Gaussian<DIM> &
      setCovariance(MatrixType &cov);
      Gaussian<DIM> &
      setMean(VectorType &mean);

      // methods
      VectorType
      draw() const;
      g_float
      pdf(VectorType x) const;

      // getter
      VectorType &
      mean();
      const MatrixType &
      getCovariance();

      //TODO: toFile method
    };

  /* helpers for operating with gaussians */

  // project a gaussian to a lower dimensional version
  template<int P_DIM, int DIM>
    Gaussian<P_DIM>
    projectGaussian(const Gaussian<DIM> &g, Eigen::VectorXd &dims);

}

// --> see gaussian_impl.hpp for implementation
#include <gaussian_mixture/impl/gaussian_impl.hpp>

#endif
