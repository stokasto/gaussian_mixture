#ifndef GMM_TYPES_H_
#define GMM_TYPES_H_

#include <limits>
#include <iostream>

#ifndef DEBUG_STREAM
#if 1 // can turn of debugging completely here
#define DEBUG_STREAM(X) { \
  std::cerr << "DEBUG: " X << std::endl; \
}
#else
#define DEBUG_STREAM(X)
#endif
#endif

#ifndef ERROR_STREAM
#if 1 // can turn of debugging completely here
#define ERROR_STREAM(X) { \
  std::cerr << "ERROR: " X << std::endl; \
}
#else
#define ERROR_STREAM(X)
#endif
#endif

namespace gmm
{
  const bool DEBUG = true;

  typedef float g_float;
  const g_float GFLOAT_MIN = std::numeric_limits<g_float>::min();//-1e7;
  const g_float GFLOAT_MAX = std::numeric_limits<g_float>::max();//1e7;

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
  // EM algorithm on gaussian mixture models
  template<int DIM>
    class EM;
}

#endif /* GMM_TYPES_H_ */
