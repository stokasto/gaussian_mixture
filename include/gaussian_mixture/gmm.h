#ifndef GMM_H_
#define GMM_H_

#include <gaussian_mixture/gaussian.h>

#include <Eigen/Core>
#include <vector>

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
          Gaussian<DIM>::VectorType>& pats, bool &changed);
      void
      updateClusters(std::vector<int> & assignments,
          const std::vector<Gaussian<DIM>::VectorType>& pats);

    public:
      // constructor and destructor
      GMM();
      virtual
      ~GMM();

      // Methods for following the named parameters paradigm
      GMM<DIM> &
      setNumStates(int num);
      GMM<DIM> &
      initRandom(const std::vector<Gaussian<DIM>::VectorType> &data);
      GMM<DIM> &
      initKmeans(const std::vector<Gaussian<DIM>::VectorType> &data, int max_iter = 120);
      GMM<DIM> &
      initUniformAlongAxis(const std::vector<Gaussian<DIM>::VectorType> &data, int axis = 0);
      GMM<DIM> &
      setMean(int state, Gaussian<DIM>::VectorType &mean);
      GMM<DIM> &
      setCovariance(int state, Gaussian<DIM>::MatrixType &cov);
      GMM<DIM> &
      setPrior(int state, g_float prior);
      GMM<DIM> &
      setPriors(Eigen::VectorXd priors);

      // general methods
      // draw a sample
      Gaussian<DIM>::VectorType
      draw() const;
      // calculate pdf for given point x
      g_float
      pdf(Gaussian<DIM>::VectorType x) const;
      // compute most likely gauss from mixture model for point x
      int
      mostLikelyGauss(Gaussian<DIM>::VectorType x) const;

      // getter
      Gaussian<DIM>::VectorType &
      getMean(int state);
      Gaussian<DIM>::MatrixType &
      getCovariance(int state);

      // TODO: toFile methods

    };
}

// --> see gmm_impl.hpp for implementation

#include <gaussian_mixture/impl/gmm_impl.hpp>

#endif /* GMM_H_ */