#ifndef DISTRIBUTIONS_H_
#define DISTRIBUTIONS_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/random.h>
// Gaussian.h holds the message definition
#ifdef GMM_ROS
#include <gaussian_mixture/GaussianModel.h>
#endif

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

  /** Definition of a DIM dimensional Gaussian distribution.
   *  @author Jost Tobias Springenberg
   */
  template<int DIM>
    class Gaussian
    {
    public:
      /** Type of the Covariance Matrix for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, DIM, DIM> MatrixType;
      /** Type of a Sample Vector for a Gaussian of DIM dimensions */
      typedef Eigen::Matrix<g_float, DIM, 1> VectorType;

    private:
      VectorType mean_;
      MatrixType covariance_;
      // preallocate cholesky decomposition here for efficiency
      Eigen::LLT<MatrixType> cholesky_;
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

      // constructors / destructors

      /** Create and instance of a Gaussian with given input dimensionality.
       *  NOTE: This Gaussian will have 0 mean and unit variance.
       *        Use setMean() and setCovariance() to change them.
       */
      Gaussian();
      virtual
      ~Gaussian();

      // Methods for following the named parameters paradigm

      /** Set the Covariance matrix of this Gaussian.
       *  This will also precompute the cholesky decomposition.
       *
       *  @param cov the covariance matrix
       *  @return returns reference to this
       */
      Gaussian<DIM> &
      setCovariance(const MatrixType &cov);

      /** Set the Mean vector of this Gaussian.
       *
       *  @param mean the mean as a vector
       *  @return returns reference to this
       */
      Gaussian<DIM> &
      setMean(const VectorType &mean);

      // methods
      /** Draw a sample from the distribution.
       *
       *  @param res the resulting sample of size DIM
       */
      void
      draw(VectorType &res) const;

      /** Calculate the pdf function at position x.
       *
       *  @param x the position for which the pdf should be computed
       */
      g_float
      pdf(const VectorType x);

      // getter
      /** Get a non-const reference to the mean vector.
       *
       * @return returns the mean of this gaussian as a DIM dimensional vector
       */
      VectorType &
      mean();

      /** Get a const reference to the covariance matrix.
       *
       * @return returns the covariance matrix of this gaussian as a DIMxDIM matrix
       */
      const MatrixType &
      getCovariance() const;

      /** Get a const reference to the mean vector.
       *
       * @return returns the mean of this gaussian as a DIM dimensional vector
       */
      const VectorType &
      getMean() const;

      /** Get the dimension of the Gaussian.
       *
       * @return returns DIM
       */
      int
      getDIM() const;

      /** Get a converter instance.
       *  The converter can be used to project the Gaussian to a lower dimensional version
       *  or to compute conditional / marginal distributions.
       *
       * @return returns a GaussianConverter that converts this Gaussian to a Gaussian of dim P_DIM
       */
      template<int P_DIM>
        GaussianConverter<DIM, P_DIM>
        getConverter() const;

#ifdef GMM_ROS
      // methods for interacting with ros

      /** Initialize the Gaussian from a ros msg.
       *
       * @return returns true if the message was filled correctly
       */
      bool
      fromMessage(const gaussian_mixture::GaussianModel &msg);

      /** Serialize the Gaussian to a ros msg.
       *
       * @return returns true if the Gaussian could be serialized correctly
       */
      bool
      toMessage(gaussian_mixture::GaussianModel &msg) const;

      /** Serialize the Gaussian to a ros bag file.
       *
       * @return returns true if the Gaussian could be serialized correctly
       */
      bool
      toBag(const std::string &bag_file);

      /** Initialize the Gaussian from a ros bag file.
       *
       * @return returns true if a GaussianModel message could be read from the bag file
       */
      bool
      fromBag(const std::string &bag_file);
#endif

      /** Serialize the Gaussian to a binary file.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the Gaussian could be serialized correctly
       */
      bool
      toBinaryFile(const std::string &fname);

      /** Read the Gaussian from a binary file.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the Gaussian could be initialized correctly
       */
      bool
      fromBinaryFile(const std::string &fname);

      /** Serialize the Gaussian to a binary stream.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the Gaussian could be serialized correctly
       */
      bool
      toStream(std::ofstream &out);

      /** Read the Gaussian from a binary stream.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the Gaussian could be initialized correctly
       */
      bool
      fromStream(std::ifstream &in);
    };

}

// --> see gaussian_impl.hpp for implementation
#include <gaussian_mixture/impl/gaussian_impl.hpp>

#endif
