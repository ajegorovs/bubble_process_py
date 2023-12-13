""""
possibly a switch for GPU support/CPU multithreading and parameters
* launch kernels * by (*) mark actions where parallelization can be beneficial
import libraries. split acess to workers.

setup case parameters

find paths to all viable images.
-------
create file system based on case parameters
------
image processing stage 1) preparation:
deal with mask. determine crop region
load images     *      
remap images    *
rotate images
crop images
-------
image processing stage 2) processing:
calculate mean * (this blocks parallelization if whole stack mean is calculated)
split stack into batches and process on GPU or CPU-parallel
--------
remove artifacts on each frame using openCV contour functions

"""