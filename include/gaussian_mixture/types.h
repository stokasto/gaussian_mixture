#ifndef GMM_TYPES_H_
#define GMM_TYPES_H_

namespace gmm
{
  typedef float g_float;

  // gaussian definition
  template<int DIM>
    class Gaussian;
  // converter definition
  template<int DIM, int P_DIM>
    class GaussianConverter;
  // gaussian mixture model definition
  template<int DIM>
    class GMM;
  // gaussian mixture regression definition
  template<int DIM, int P_DIM>
    class GMR;
}

#endif /* GMM_TYPES_H_ */
