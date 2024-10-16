#!/bin/bash

#
# This script is intended to be used inside the ROCm-6.2 build agent Singularity container
# to build WarpX for use in a Singularity WarpX container.
#

$WITH_CONDA
cmake --preset $PRESET -B $BUILD_DIR
cmake --build $BUILD_DIR --target lib_2d --target lib_3d --target pyamrex_pip_wheel --target pip_wheel

# Make a copy of some missing pieces in the ROCm libraries provided by `singularity --rocm`
if [ ! -d $BUILD_DIR/hip ]; then
    mkdir $BUILD_DIR/hip
fi
cp /opt/rocm/lib/libamdhip64.so.6.?.* $BUILD_DIR/hip
cp /opt/rocm/lib/librocprofiler-register.so.0.?.* $BUILD_DIR/hip
cp /opt/amdgpu/share/libdrm/amdgpu.ids $BUILD_DIR/hip
