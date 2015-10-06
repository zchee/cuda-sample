#!/bin/bash
nvcc cdp_lu.cu cdp_lu_main.cu dgetrf.cu dgetf2.cu dlaswp.cu -lcublas_device -lcublas -lcudadevrt  -arch=sm_35 -rdc=true -Xcompiler  -fopenmp -lgomp -o cdpLU
