#ifndef GMM_IMPL_HPP_
#define GMM_IMPL_HPP_

#include <cmath>

#include <gaussian_mixture/gmm.h>

namespace gmm
{
  template<int DIM>
    GMM<DIM>::GMM() :
      initialized_(false), num_states_(0)
    {
    }

  template<int DIM>
    GMM<DIM>::~GMM()
    {
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::setNumStates(int num)
    {
      // allocate appropriate number of gaussians
      gaussians_.resize(num);
      // set uniform priors
      priors_.setConstant(1. / num);
      // finally store number of states for convenience
      num_states_ = num;
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::initRandom(const std::vector<typename Gaussian<DIM>::VectorType> &data)
    {
      // pick random means from the data
      for (int i = 0; i < num_states_; ++i)
        {
          int next = rand() % data.size();
          Gaussian<DIM> &g = gaussians_[i];
          // adapt mean --> covariance is left to be the identity
          g.mean() = data[next];
        }
      initialized_ = true;
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::initKmeans(const std::vector<typename Gaussian<DIM>::VectorType> &data, int max_iter)
    {
      std::vector<int> assignments(data.size());
      std::vector<int> assignments2(data.size());
      bool changed = false;
      // set old assignments to -1 initially
      for (int i = 0; i < assignments2.size(); ++i)
        {
          assignments2[i] = -1;
        }
      // first init randomly
      initRandom(data);

      // initial cluster step
      cluster(assignments, assignments2, data, changed);
      changed = true;

      // then cluster for the remaining number of iterations
      for (int iter = 1; iter < max_iter; ++iter)
        {
          // 1) --> update clusters with the new assignments
          if (iter % 2 != 0)
            {
              updateClusters(assignments, data);
            }
          else
            {
              updateClusters(assignments2, data);
            }

          // 2) --> cluster step
          // be sure to swap assignments and old_assignments every 2nd iteration
          if (iter % 2 == 0)
            {
              cluster(assignments, assignments2, data, changed);
            }
          else
            {
              cluster(assignments2, assignments, data, changed);
            }
          // check if an assignment changed --> if not we are done
          if (!changed)
            break;
        }

      initialized_ = true;
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::initUniformAlongAxis(const std::vector<typename Gaussian<DIM>::VectorType> &data,
        int axis)
    {
      assert(axis >= 0 && axis < DIM);
      // first calculate mean along the selected axis
      g_float min = 1e7;
      g_float max = -1e7;
      for (int i = 0; i < int(data.size()); ++i)
        {
          g_float tmp = data[i](axis);
          if (tmp < min)
            min = tmp;
          if (tmp > max)
            max = tmp;
        }
      // next init gaussians to those data points that are closest
      // to a uniform distribution along the axis
      for (int i = 0; i < num_states_; ++i)
        {
          // calculate desired value
          g_float desired = (max - min) * i / num_states_ + min;
          g_float best_dist = 1e7;
          int best = 0;
          // find data point closest to this one
          for (int j = 0; j < int(data.size()); ++j)
            {
              g_float dist = desired - data[j](axis);
              if (dist < best_dist)
                {
                  best_dist = dist;
                  best = j;
                }
            }
          // adapt mean --> covariance is left to be the identity
          gaussians_[i].setMean(data[best]);
        }
      initialized_ = true;
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::setMean(int state, typename Gaussian<DIM>::VectorType &mean)
    {
      assert(state >= 0 && state < num_states_);
      gaussians_[state].setMean(mean);
      return *this;

    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::setCovariance(int state, typename Gaussian<DIM>::MatrixType &cov)
    {
      assert(state >= 0 && state < num_states_);
      gaussians_[state].setCovariance(cov);
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::setPrior(int state, g_float prior)
    {
      assert(state >= 0 && state < num_states_);
      priors_[state] = prior;
      return *this;
    }

  template<int DIM>
    GMM<DIM> &
    GMM<DIM>::setPriors(Eigen::VectorXd prior)
    {
      assert(prior.size() == num_states_);
      priors_ = prior;
      return *this;
    }

  template<int DIM>
    g_float
    GMM<DIM>::cluster(std::vector<int> &assignments, std::vector<int> &old_assignments,
        const std::vector<typename Gaussian<DIM>::VectorType>& pats, bool &changed)
    {
      int best_idx = 0;
      g_float summed_dist;
      changed = false; // initially assume no assignment changed

      //#pragma omp parallel for
      for (int i = 0; i < (int) pats.size(); ++i)
        { // for each pattern find the best assignment to a prototype
          g_float dist = 1e7;
          // search through all cluster centers
          for (int g_idx = 0; g_idx < (int) gaussians_.size(); ++g_idx)
            {
              g_float tmp_dist = (pats[i] - gaussians_[g_idx].mean()).squaredNorm();
              if (tmp_dist < dist)
                {
                  dist = tmp_dist;
                  best_idx = g_idx;
                }
            }
          //ROS_DEBUG_STREAM(KM_LVL, << "Best prototype for: " << pats[i].transpose()
          //    << " is: " << centers_[best_idx]->transpose() << " at idx: " << best_idx);
          /*if (!changed // we have not  yet found and assignment that has changed
           && assignments[i] != best_idx)
           { // check if assignment changed --> if so update changed flag
           changed = true;
           }*/
          assignments[i] = best_idx;
          summed_dist += dist;
        }
      for (int i = 0; i < (int) pats.size(); ++i)
        {
          if (assignments[i] != old_assignments[i])
            {
              changed = true;
              break;
            }
        }
      //ROS_DEBUG_STREAM(KM_LVL, << "DONE COMPUTING CLUSTERS");
      return summed_dist;
    }

  template<int DIM>
    void
    GMM<DIM>::updateClusters(std::vector<int> & assignments, const std::vector<typename Gaussian<
        DIM>::VectorType>& pats)
    {
      int patterns_per_cluster[num_states_];
      int curr_assignment = 0;
      int prev_count = 0;

      for (int i = 0; i < num_states_; i++)
        { // reset pattern per cluster counts
          patterns_per_cluster[i] = 0;
        }

      for (int i = 0; i < (int) pats.size(); i++)
        { // traverse all training patterns
          curr_assignment = assignments[i];
          // update number of patterns per cluster
          ++patterns_per_cluster[curr_assignment];
          // update center of closest prototype
          gaussians_[curr_assignment].mean() += pats[i];
          //ROS_DEBUG_STREAM(KM_LVL+1, << "prototype " << curr_assignment << " has now "
          //    << patterns_per_cluster[curr_assignment] << " assigned patterns");
        }

      //#pragma omp parallel for
      for (int i = 0; i < num_states_; i++)
        { // normalize all prototype positions to get the proper mean of the pattern vectors
          if (patterns_per_cluster[i] > 0.)
            { // beware of the evil division by zero :)
              gaussians_[curr_assignment].mean
                  /= (patterns_per_cluster[i] > 0) ? patterns_per_cluster[i] : 1.;
            }
        }
      //ROS_DEBUG("DONE updating kmeans");
    }

  template<int DIM>
    typename Gaussian<DIM>::VectorType
    GMM<DIM>::draw() const
    {
      int state = 0;
      g_float accum = 0.;
      g_float thresh = random_uniform_0_1();
      // first accumulate priors until thresh is reached
      while ((accum < thresh) && (state < num_states_))
        {
          accum += priors_[state];
          ++state;
        }
      // finally draw from the distribution belonging to the selected state
      return gaussians_[state].draw();
    }

  template<int DIM>
    g_float
    GMM<DIM>::pdf(typename Gaussian<DIM>::VectorType x) const
    {
      g_float likeliehood = 0., tmp = 0.;
      for (int i = 0; i < num_states_; ++i)
        {
          // multiply individual pdf with prior
          tmp = priors_[i] * gaussians_[i].pdf(x);
          // add to result
          likeliehood += tmp;
        }
      return likeliehood;
    }

  template<int DIM>
    int
    GMM<DIM>::mostLikelyGauss(typename Gaussian<DIM>::VectorType x) const
    {
      g_float best_likeliehood = 0., tmp = 0.;
      int best = 0;
      for (int i = 0; i < num_states_; ++i)
        {
          // multiply individual pdf with prior
          tmp = priors_[i] * gaussians_[i].pdf(x);
          if (tmp > best_likeliehood)
            {
              best_likeliehood = tmp;
              best = i;
            }
        }
      return best;
    }

  template<int DIM>
    typename Gaussian<DIM>::VectorType &
    GMM<DIM>::getMean(int state)
    {
      assert(state >= 0 && state < num_states_);
      return gaussians_[state].mean();
    }

  template<int DIM>
    typename Gaussian<DIM>::MatrixType &
    GMM<DIM>::getCovariance(int state)
    {
      assert(state >= 0 && state < num_states_);
      return gaussians_[state].getCovariance();
    }
}

#endif /* GMM_IMPL_HPP_ */
