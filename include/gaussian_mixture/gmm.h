#ifndef GMM_H_
#define GMM_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>
// include ros message
#ifdef GMM_ROS
#include <gaussian_mixture/GaussianMixtureModel.h>
#endif

#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <fstream>

#ifdef GMM_ROS
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#endif

namespace gmm
{

  /** Definition of a DIM dimensional Gaussian Mixture Model.
   *  @author Jost Tobias Springenberg
   */
  template<int DIM>
    class GMM
    {
    private:
      /** just for convenience --> must always be == gaussians_.size() */
      int num_states_;
      /** bool denoting whether this gmm was initialized either by any of the init methods */
      bool initialized_;
      /** vector containing all gaussians contributing to the gaussian mixture model */
      std::vector<Gaussian<DIM> > gaussians_;
      /** vector containing the priors of all gaussians */
      Eigen::VectorXd priors_;

      /* methods for kmeans clustering */

      /** Assign one gaussian mean to each vector from the training data.
       *
       *  @param assignments this will store the new assignments for each training pattern
       *  @param old_assignments this should be filled with the assignments from the last call
       *         used for checking whether any assignment changed
       *  @param pats the training patterns
       *  @changed denotes whether the assignment for at least one training pattern was changed
       *  @return returns the summed distance of all training patterns to their assigned mean vector
       */
      g_float
      cluster(std::vector<int> &assignments, std::vector<int> &old_assignments, const std::vector<
          typename Gaussian<DIM>::VectorType>& pats, bool &changed);

      /** Update mean and variance of all gaussians according to the training data.
       *
       * @param assignments the assignments for all training patterns
       * @param pats the training patterns
       */
      void
      updateClusters(std::vector<int> & assignments, const std::vector<
          typename Gaussian<DIM>::VectorType>& pats);

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      // constructor and destructor

      /** Create a new Gaussian Mixture model instance.
       *  NOTE: it is necessary to initialize the gmm using any of the init methods.
       */
      GMM();
      virtual
      ~GMM();

      // Methods for following the named parameters paradigm

      /** Set the number of states == number of gaussians that the gmm should consist of.
       *
       * @param num the number of states
       * @return returns a reference to this
       */
      GMM<DIM> &
      setNumStates(int num);

      /** Initialize the mean of each gaussian to a random data point from the trainint data.
       *  NOTE: All gaussians will have unit variance (the Identity matrix)
       *
       * @param data the vector of training data
       * @return returns a reference to this
       */
      GMM<DIM> &
      initRandom(const std::vector<typename Gaussian<DIM>::VectorType> &data);

      /** Initialize each gaussian using kmeans clustering.
       * This will initialize the mean and covariance of each gaussian.
       *
       * @param data the vector of training data
       * @return returns a reference to this
       */
      GMM<DIM> &
      initKmeans(const std::vector<typename Gaussian<DIM>::VectorType> &data, int max_iter = 120);

      /** Initialize the mean of all gaussian by distributing them uniformly along one axis.
       *  NOTE: All gaussians will have unit variance (the Identity matrix)
       *
       * @param data the vector of training data
       * @return returns a reference to this
       */
      GMM<DIM> &
      initUniformAlongAxis(const std::vector<typename Gaussian<DIM>::VectorType> &data, int axis =
          0);

      /** Set the mean of the nth gaussian.
       *
       * @param state the gaussian for which the mean should be changed
       * @param mean the new mean
       * @return returns a reference to this
       */
      GMM<DIM> &
      setMean(int state, typename Gaussian<DIM>::VectorType &mean);

      /** Set the covariance of the nth gaussian.
       *
       * @param state the gaussian for which the covariance should be changed
       * @param covariance the new covariance matrix of dimensions DIMxDIM
       * @return returns a reference to this
       */
      GMM<DIM> &
      setCovariance(int state, typename Gaussian<DIM>::MatrixType &cov);\

      /** Set the prior of the nth gaussian.
       *
       * @param state the gaussian for which the covariance should be changed
       * @param prior the new prior probability
       * @return returns a reference to this
       */
      GMM<DIM> &
      setPrior(int state, g_float prior);

      /** Set the priors of the all gaussians.
       *
       * @param priors a num_states dimensional vector containing the new priors.
       * @return returns a reference to this
       */
      GMM<DIM> &
      setPriors(Eigen::VectorXd priors);

      // general methods
      /** Draw a sample from the gmm
       *
       * @param result a vector that will contain the resulting sample.
       */
      void
      draw(typename Gaussian<DIM>::VectorType &result) const;

      /** Calculate the pdf function at position x.
       *
       *  @param x the position for which the pdf should be computed
       */
      g_float
      pdf(const typename Gaussian<DIM>::VectorType x) const;

      /** Compute most likely gauss from mixture model for point x.
       *
       *  @param x the position for which the most likely gaussian should be computed
       *  @return returns the position in the gaussians_ array of the most likely gaussian
       */
      int
      mostLikelyGauss(const typename Gaussian<DIM>::VectorType x) const;

      // getter
      /** Get a GMR instance.
       *  The Regression Model can be used to draw from conditional or marginal distributions.
       *
       * @return returns a GMR that regresses this GMM to the last P_DIM dimensions
       */
      template<int P_DIM>
        GMR<DIM, P_DIM>
        getRegressionModel() const;

      /** Get a EM instance.
       *  The EM instance can be used to adapt the gaussian means and covariance matrices using
       *  the em algorithm.
       *
       *  @return returns an EM instance
       */
      EM<DIM>
      getEM();

      /** Get a const reference to the nth gaussian.
       *
       * @param state the position of the requested gaussian in the gaussians_ array
       * @return returns the nth gaussian
       */
      const Gaussian<DIM> &
      getGaussian(int state) const;

      /** Get a non-const reference to the nth gaussian.
       *
       * @param state the position of the requested gaussian in the gaussians_ array
       * @return returns the nth gaussian
       */
      Gaussian<DIM> &
      gaussian(int state);

      /** Get a non-const reference to the mean of the nth gaussian.
       *
       * @param state the position of the requested gaussian in the gaussians_ array
       * @return returns the mean of the nth gaussian
       */
      typename Gaussian<DIM>::VectorType &
      getMean(int state);

      /** Get a non-const reference to the covariance matrix of the nth gaussian.
       *
       * @param state the position of the requested gaussian in the gaussians_ array
       * @return returns the covariance matrix of the nth gaussian
       */
      typename Gaussian<DIM>::MatrixType &
      getCovariance(int state);

      /** Get the number of states.
       *
       * @return the number of states (number of gaussians) that the gmm consists of.
       */
      int
      getNumStates() const;

      /** Get the prior probability of the nth gaussian.
       *
       * @param state the position of the requested gaussian in the gaussians_ array
       * @return returns the prior probability of the nth gaussian
       */
      g_float
      getPrior(int state) const;

      /** Get a const reference to all prior probabilities.
       *
       * @return returns all prior probabilities as a VectorXd
       */
      const Eigen::VectorXd
      getPriors() const;

      /** Force the gmm to be marked as initialized
       *  WARNING: Calling this method should never be necessary outside of tests.
       */
      void
      forceInitialize();

#ifdef GMM_ROS
      // methods for interacting with ros

      /** Initialize the gmm from a ros msg.
       *
       * @return returns true if the message was filled correctly
       */
      bool
      fromMessage(const gaussian_mixture::GaussianMixtureModel &msg);

      /** Serialize the gmm to a ros msg.
       *
       * @return returns true if the gmm could be serialized correctly
       */
      bool
      toMessage(gaussian_mixture::GaussianMixtureModel &msg) const;

      /** Serialize the gmm to a ros bag file.
       *
       * @return returns true if the gmm could be serialized correctly
       */
      bool
      toBag(const std::string &bag_file);

      /** Initialize the gmm from a ros bag file.
       *
       * @return returns true if a GaussianMixtureModel message could be read from the bag file
       */
      bool
      fromBag(const std::string &bag_file);
#endif
      /** Serialize the gmm to a binary file.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the gmm could be serialized correctly
       */
      bool
      toBinaryFile(const std::string &fname);

      /** Read the gmm from a binary file.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the gmm could be initialized correctly
       */
      bool
      fromBinaryFile(const std::string &fname);

      /** Write the gmm to a binary stream.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the gmm could be serialized correctly
       */
      bool
      toStream(std::ofstream &out);

      /** Read the gmm from a binary stream.
       *  NOTE: this is not architecture independant.
       *
       * @return returns true if the gmm could be initialized correctly
       */
      bool
      fromStream(std::ifstream &in);

    };
}

// --> see gmm_impl.hpp for implementation
#include <gaussian_mixture/impl/gmm_impl.hpp>

#endif /* GMM_H_ */
