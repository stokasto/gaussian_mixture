cmake_minimum_required(VERSION 2.4.6)

if(${GMM_NO_ROS})
include(CMakeLists.noros.txt)
else(${GMM_NO_ROS})
include(CMakeLists.ros.txt)
endif(${GMM_NO_ROS})

