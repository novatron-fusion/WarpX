
Building WarpX with ROCm on LUMI
================================

```
git clone https://github.com/ECP-WarpX/WarpX WarpX-upstream
cd WarpX-upstream
mkdir build-tools
tar xvzf build-tools.tar.gz
```

Next, create a container for a build-agent capable of building WarpX with ROCm 6.2:
```
singularity build --fakeroot build-tools/rocm-6.2-build-agent.sif build-tools/rocm-6.2-build-agent.def
```

Now use this build agent to compile WarpX for LUMI:
```
singularity exec --env BUILD_DIR=build_lumi --env PRESET=lumi --containall --bind $(pwd) --cwd $(pwd) build-tools/rocm-6.2-build-agent.sif build-tools/compile-rocm.sh
```

Build extra packages to be installed and put their wheel files in $BUILD_DIR/whl_extras. For example, to build an AMD ROCm version of CuPy, do this:

```
singularity exec --rocm --no-mount hostfs --env HCC_AMDGPU_TARGET=gfx90a --env CUPY_INSTALL_USE_HIP=1 --env ROCM_HOME=/opt/rocm build-tools/rocm-6.2-build-agent.sif sh -c '$WITH_CONDA; pip wheel cupy -w build_lumi/whl_extras'
```

Finally, package the WarpX build as a singularity container to be used on LUMI:
```
singularity build --fakeroot --build-arg build_dir=build_lumi --build-arg remove_mpi=true -F build-tools/warpx.sif build-tools/rocm-singularity.def
```

This should end with
```
INFO:    Adding environment to container
INFO:    Adding runscript
INFO:    Creating SIF file...
INFO:    Build complete: build-tools/warpx.sif
```

If you want to use the image without rebinding the MPI libraries to use the host libraries, don't remove the installed mpi libraries inside the container:

```
singularity build --fakeroot --build-arg build_dir=build_lumi --build-arg remove_mpi=false -F build-tools/warpx.sif build-tools/rocm-singularity.def
```

Upload the container and ... it is possible to run the simulation through the python interface:

```
singularity run --rocm --no-mount hostfs --bind multipole-mirror-testcase:/opt/run_sim --env SIMULATION_CMD="python /opt/run_sim/run_sim.py" build-tools/warpx-lumi.sif
```

Or directly in the WarpX app build:

```
singularity run --rocm --no-mount hostfs --bind multipole-mirror-testcase:/opt/run_sim --env SIMULATION_CMD="warpx.3d /opt/run_sim/multipole-mirror.warpx" build-tools/warpx-lumi.sif
```

To run it under 

```
singularity run --rocm --no-mount hostfs --bind multipole-mirror-testcase:/opt/run_sim --env SIMULATION_CMD="valgrind --tool=massif --threshold=0.01 --time-unit=ms --massif-out-file=massif.%p.out python /opt/run_sim/run_sim.py" build-tools/warpx-lumi.sif
```

Or directly in the WarpX app build:

```
singularity run --rocm --no-mount hostfs --bind multipole-mirror-testcase:/opt/run_sim --env SIMULATION_CMD="valgrind --tool=massif --threshold=0.01 --time-unit=ms --massif-out-file=massif.%p.out warpx.3d /opt/run_sim/multipole-mirror.warpx" build-tools/warpx-lumi.sif
```


