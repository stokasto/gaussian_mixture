#ifndef GMM_H_
#define GMM_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>
// include ros message
#include <gaussian_mixture/GaussianMixtureModel.h>

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
  template<int DIM>
    class GMM
    {
    private:
      int num_states_; // just for convenience --> must always be == gaussians_.size()
      bool initialized_;
      std::vector<Gaussian<DIM> > gaussians_;
      Eigen::VectorXd priors_;

      /* methods for kmeans clustering */
      g_float
      cluster(std::vector<int> &assignments, std::vector<int> &old_assignments, const std::vector<
          typename Gaussian<DIM>::VectorType>& pats, bool &changed);
      void
      updateClusters(std::vector<int> & assignments, const std::vector<
          typename Gaussian<DIM>::VectorType>& pats);

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      // constructor and destructor
      GMM();
      virtual
      ~GMM();

      // Methods for following the named parameters paradigm
      GMM<DIM> &
      setNumStates(int num);
      GMM<DIM> &
      initRandom(const std::vector<typename Gaussian<DIM>::VectorType> &data);
      GMM<DIM> &
      initKmeans(const std::vector<typename Gaussian<DIM>::VectorType> &data, int max_iter = 120);
      GMM<DIM> &
      initUniformAlongAxis(const std::vector<typename Gaussian<DIM>::VectorType> &data, int axis =
          0);
      GMM<DIM> &
      setMean(int state, typename Gaussian<DIM>::VectorType &mean);
      GMM<DIM> &
      setCovariance(int state, typename Gaussian<DIM>::MatrixType &cov);
      GMM<DIM> &
      setPrior(int state, g_float prior);
      GMM<DIM> &
      setPriors(Eigen::VectorXd priors);

      // general methods
      // draw a sample
      void
      draw(typename Gaussian<DIM>::VectorType &result) const;
      // calculate pdf for given point x
      g_float
      pdf(const typename Gaussian<DIM>::VectorType x) const;
      // compute most likely gauss from mixture model for point x
      int
      mostLikelyGauss(const typename Gaussian<DIM>::VectorType x) const;

      // getter
      template<int P_DIM>
        GMR<DIM, P_DIM>
        getRegressionModel() const;
      EM<DIM>
      getEM();

      const Gaussian<DIM> &
      getGaussian(int state) const;
      Gaussian<DIM> &
      gaussian(int state);

      typename Gaussian<DIM>::VectorType &
      getMean(int state);

      typename Gaussian<DIM>::MatrixType &
      getCovariance(int state);

      int
      getNumStates() const;

      g_float
      getPrior(int state) const;

      const Eigen::VectorXd
      getPriors() const;

      void
      forceInitialize();

#ifdef GMM_ROS
      bool
      fromMessage(const gaussian_mixture::GaussianMixtureModel &msg);
      bool
      toMessage(gaussian_mixture::GaussianMixtureModel &msg) const;
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

// --> see gmm_impl.hpp for implementation
#include <gaussian_mixture/impl/gmm_impl.hpp>

#endif /* GMM_H_ */
