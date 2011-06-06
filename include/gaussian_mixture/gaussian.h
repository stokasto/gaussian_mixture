#ifndef DISTRIBUTIONS_H_
#define DISTRIBUTIONS_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/random.h>
// Gaussian.h holds the message definition
#include <gaussian_mixture/GaussianModel.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <string>
#include <fstream>

#ifdef GMM_ROS
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#endif

namespace gmm
{

  /* definition of a n dimensional gaussian */
  template<int DIM>
    class Gaussian
    {
    public:
      typedef Eigen::Matrix<g_float, DIM, DIM> MatrixType;
      typedef Eigen::Matrix<g_float, DIM, 1> VectorType;

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
      int
      getDIM() const;

      template<int P_DIM>
        GaussianConverter<DIM, P_DIM>
        getConverter() const;

#ifdef GMM_ROS
      bool
      fromMessage(const gaussian_mixture::GaussianModel &msg);
      bool
      toMessage(gaussian_mixture::GaussianModel &msg) const;
      bool
      toBag(const std::string &bag_file);
      bool
      fromBag(const std::string &bag_file);
#endif
      bool
      toBinaryFile(const std::string &fname);
      bool
      fromBinaryFile(const std::string &fname);
      bool
      toStream(std::ofstream &out);
      bool
      fromStream(std::ifstream &in);
    };

}

// --> see gaussian_impl.hpp for implementation
#include <gaussian_mixture/impl/gaussian_impl.hpp>

#endif
