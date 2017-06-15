# PErasure
A parallel Cauchy Reed-Solomon Coding Library for GPUs. 
Implementation of this [paper](http://ieeexplore.ieee.org/document/7248360/)


## Installation
Clone the repository. There is no need to install the project.

### Dependencies
Install the following libraries:
```
git clone https://github.com/tsuraan/Jerasure.git
git clone http://lab.jerasure.org/jerasure/gf-complete.git
```
Follow the instructions [here](https://github.com/tsuraan/Jerasure) to correctly install the two libraries.

## Compilation
Compile PErasure with:
```
nvcc perasure.cu -o bin/output_name -I /usr/local/include/jerasure -lJerasure -lgf_complete
```

Compile sequential jerasure with:
```
g++ seq_erasure.cpp -o bin/output_name -I /usr/local/include/jerasure -lJerasure -lgf_complete
```

## Running
In order to run you need to pass the following arguments:
```
k: number of data devices
m: number of coding devices
w: number of rows. The following must hold: m + k <= 2^w
size: how many bytes per row

example:
./perasure 3 4 8 8
```

## Performance evaluation
There are a series of scripts in the bin folder which can be used to perform a series of runs of the algorithm with a given configuration.
Modify those if you want to test different configuration.
Note that perasure_mem.cu is used to test performances taking into account the overhead due to data copy to and from the device.

