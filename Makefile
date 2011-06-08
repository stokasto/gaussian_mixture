ifndef GMM_NO_ROS
include $(shell rospack find mk)/cmake.mk
else
all:
	rm -rf build
	mkdir build
	cd build; cmake .. -DGMM_NO_ROS=1 && make
endif
