#ifndef GAUSSIAN_IMPL_HPP_
#define GAUSSIAN_IMPL_HPP_

#include <gaussian_mixture/gaussian.h>

namespace gmm
{

  /* gaussian implementation */
  template<int DIM>
    Gaussian<DIM>::Gaussian() :
      mean_(VectorType::Zero()), covariance_(MatrixType::Identity()), cholesky_(covariance_.llt()),
          partition_(sqrt(pow(2 * M_PI, DIM)))
    {
    }

  template<int DIM>
    Gaussian<DIM>::~Gaussian()
    {
    }

  template<int DIM>
    void
    Gaussian<DIM>::draw(typename Gaussian<DIM>::VectorType &res) const
    {
      // at first draw N random variables from a normal distribution
      for (int i = 0; i < DIM; ++i)
        res(i) = random_normal();
      // multiply with covariance matrix and add mean
      // to get a proper sample from the current distribution
      res = cholesky_.matrixL() * res + mean_;
    }

  template<int DIM>
    g_float
    Gaussian<DIM>::pdf(const Gaussian<DIM>::VectorType x)
    {
      // precompute distance to mean
      tmp_ = x - mean_;
      // next compute (x - mean) * sigma^-1 * (x - mean)
      // using the cholesky decomposition
      beta_ = cholesky_.matrixL().solve(tmp_);
      alpha_ = cholesky_.matrixL().transpose().solve(beta_);
      g_float res = tmp_.dot(alpha_);
      // finally calculate pdf response
      res *= 0.5;
      res = exp(-res) / partition_;
      //if (res == 0.)
      //  res = 1e-9;
      return res;
    }

  template<int DIM>
    typename Gaussian<DIM>::VectorType &
    Gaussian<DIM>::mean()
    {
      return mean_;
    }

  template<int DIM>
    const typename Gaussian<DIM>::MatrixType &
    Gaussian<DIM>::getCovariance() const
    {
      return covariance_;
    }

  template<int DIM>
    const typename Gaussian<DIM>::VectorType &
    Gaussian<DIM>::getMean() const
    {
      return mean_;
    }

  template<int DIM>
    int
    Gaussian<DIM>::getDIM() const
    {
      return DIM;
    }

  template<int DIM>
    Gaussian<DIM> &
    Gaussian<DIM>::setCovariance(const typename Gaussian<DIM>::MatrixType &cov)
    {
      g_float tmp = 1.;
      covariance_ = cov;
      // precompute cholesky
      cholesky_.compute(covariance_);
      // TODO: assert that cov is actually symmetric positive definite
      // recompute partition
      tmp = covariance_.determinant();
      partition_ = sqrt(pow(2 * M_PI, DIM) * tmp);
      return *this;
    }

  template<int DIM>
    Gaussian<DIM> &
    Gaussian<DIM>::setMean(const typename Gaussian<DIM>::VectorType &mean)
    {
      mean_ = mean;
      return *this;
    }

  template<int DIM>
    template<int P_DIM>
      GaussianConverter<DIM, P_DIM>
      Gaussian<DIM>::getConverter() const
      {
        return GaussianConverter<DIM, P_DIM> ().setInputGaussian(*this);
      }

  template<int DIM>
    bool
    Gaussian<DIM>::toBinaryFile(const std::string &fname)
    {
      bool res = true;
      std::ofstream out(fname.c_str(), std::ios_base::binary);
      res &= toStream(out);
      out.close();
      return res;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::toStream(std::ofstream &out)
    {
      int dim = DIM;
      out.write((char*) (&dim), sizeof(int));
      out.write((char*) (&mean_), sizeof(VectorType));
      out.write((char*) (&covariance_), sizeof(MatrixType));
      return true;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::fromBinaryFile(const std::string &fname)
    {
      std::ifstream in;
      bool res = true;
      in.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

      try
        {
          in.open(fname.c_str(), std::ios_base::binary);
          res &= fromStream(in);
        }
      catch (std::ifstream::failure& e)
        {
          ERROR_STREAM(
              (boost::format("Failed to load Gaussian Model from file '%s'") % fname).str());
          in.close();
          return false;
        }
      in.close();
      return res;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::fromStream(std::ifstream &in)
    {
      int dim;
      VectorType mean;
      MatrixType covariance;
      in.read((char*) (&dim), sizeof(int));
      if (dim != DIM)
        {
          ERROR_STREAM("called Gaussian.fromBinaryFile() with message of invalid dimension: "
              << dim << " this dim: " << DIM);
          return false;
        }
      in.read((char*) (&mean), sizeof(VectorType));
      in.read((char*) (&covariance), sizeof(MatrixType));
      setMean(mean);
      setCovariance(covariance);
      return true;
    }

#ifdef GMM_ROS

  template<int DIM>
    bool
    Gaussian<DIM>::fromMessage(const gaussian_mixture::GaussianModel &msg)
    {
      if (msg.dim != DIM)
        {
          ERROR_STREAM("called fromMessage with message of invalid dimension: " << msg.dim
              << " this dim: " << DIM);
          return false;
        }
      VectorType mean;
      MatrixType covariance;
      // copy mean from message
      for (int i = 0; i < DIM; ++i)
        mean(i) = msg.mean[i];

      // copy covariance matrix from message
      int idx = 0;
      for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
          {
            covariance(i, j) = msg.covariance[idx];
            ++idx;
          }
      setMean(mean);
      setCovariance(covariance);
      return true;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::toMessage(gaussian_mixture::GaussianModel &msg) const
    {
      msg.dim = DIM;

      // copy mean into message
      msg.mean.resize(DIM);
      for (int i = 0; i < DIM; ++i)
        msg.mean[i] = mean_(i);

      // copy covariance matrix into message in row major form
      msg.covariance.resize(DIM * DIM);
      int idx = 0;
      for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
          {
            msg.covariance[idx] = covariance_(i, j);
            ++idx;
          }
      return true;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::toBag(const std::string &bag_file)
    {
      ros::Time::init();
      try
        {
          rosbag::Bag bag(bag_file, rosbag::bagmode::Write);
          gaussian_mixture::GaussianModel msg;
          if (!toMessage(msg))
            {
              ERROR_STREAM("Could not convert Gaussian to message.");
              return false;
            }
          bag.write("gaussian", ros::Time::now(), msg);
          bag.close();
        }
      catch (rosbag::BagIOException e)
        {
          ROS_ERROR("Could not open bag file %s: %s", bag_file.c_str(), e.what());
          return false;
        }
      return true;
    }

  template<int DIM>
    bool
    Gaussian<DIM>::fromBag(const std::string &bag_file)
    {
      try
        {
          rosbag::Bag bag(bag_file, rosbag::bagmode::Read);
          rosbag::View view(bag, rosbag::TopicQuery("gaussian"));
          int count = 0;
          BOOST_FOREACH(rosbag::MessageInstance const msg, view)
{          if (count > 1)
            {
              ERROR_STREAM("More than one Gaussian stored in bag file!");
              return false;
            }
          ++count;

          gaussian_mixture::GaussianModelConstPtr model = msg.instantiate<
          gaussian_mixture::GaussianModel> ();
          if (!fromMessage(*model))
            {
              ERROR_STREAM("Could not initialize Gaussian from message!");
              return false;
            }
        }
      bag.close();
    }
  catch (rosbag::BagIOException e)
    {
      ROS_ERROR("Could not open bag file %s: %s", bag_file.c_str(), e.what());
      return false;
    }
  return true;
}

#endif

}

#endif /* GAUSSIAN_IMPL_HPP_ */
