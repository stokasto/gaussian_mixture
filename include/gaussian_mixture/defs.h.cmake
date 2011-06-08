// will be defined using cmake
#define @ROS_SWITCH@

#ifdef GMM_ROS
#include <ros/console.h>
#endif

#ifndef DEBUG_STREAM
#if 1 // can turn of debugging completely here
#ifdef GMM_ROS
#define DEBUG_STREAM ROS_DEBUG_STREAM
#else
#define DEBUG_STREAM(X) { \
    std::cerr << "DEBUG: " << X << std::endl; \
  }
#endif
#else
#define DEBUG_STREAM(X)
#endif
#endif

#ifndef ERROR_STREAM
#if 1 // can turn of debugging completely here
#ifdef GMM_ROS
#define ERROR_STREAM ROS_ERROR_STREAM
#else
#define ERROR_STREAM(X) { \
  std::cerr << "ERROR: " << X << std::endl; \
}
#endif
#else
#define ERROR_STREAM(X)
#endif
#endif

