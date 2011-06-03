#ifndef GMR_H_
#define GMR_H_

#include <gaussian_mixture/types.h>
#include <gaussian_mixture/gaussian.h>
#include <gaussian_mixture/gmm.h>

#include <Eigen/Core>

namespace gmm
{
  template<int DIM, int P_DIM>
    class GMR
    {
    private:
      // pointer to the input model
      const GMM<DIM> *model_;
      // converters for all gaussians
      std::vector<GaussianConverter<DIM, P_DIM> > converter_;
      // storage for marginal and conditional distributions extracted from the model
      std::vector<Gaussian<DIM> > marginalGaussians_;
      std::vector<Gaussian<DIM> > condGaussians_;
      // for convenience --> should be == condGaussians_.size()
      int num_states_;
      Eigen::VectorXd weights_;
      bool initialized_;

    public:
      GMR();
      virtual
      ~GMR();

      GMR<DIM, P_DIM> &
      setInputGMM(const GMM<DIM> &model);

      void
      query(const typename Gaussian<DIM>::VectorType &input,
          const typename Gaussian<P_DIM>::VectorType &sample);
      void
      getConditionalDistribution(const typename Gaussian<DIM>::VectorType &input,
          Gaussian<P_DIM> &result) const;
      const GMM<DIM> &
      getInputGMM() const;

    };
}

// implementation in gmr_impl.hpp
#include <gaussian_mixture/impl/gmr_impl.hpp>

#endif /* GMR_H_ */
