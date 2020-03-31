# Jetson-Nano-Experiments

To get the Jetson Nano to run the Intel Realsense camera is a multi-step process.

Part 1: Camera drivers
1. Navigate to https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
2. Follow the steps to register the public key and add the server to the list of repositories
3. sudo apt install librealsense2-dev (others either won't work or are unneeded)
4. plug the camera in and test with `realsense-viewer`

Part 2: Python wrapper
1. Clone the repo: https://github.com/IntelRealSense/librealsense/
2. Open this in a tab https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
3. sudo apt install libxcursor-dev libxinerama-dev python3-dev
4. cd librealsense
5. mkdir build
6. cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python
7. make -j3
8. sudo make install
9. export PYTHONPATH=$PYTHONPATH:/usr/local/lib

To get Numba, Numpy, and so on, follow this: https://github.com/jefflgaol/Install-Packages-Jetson-ARM-Family


