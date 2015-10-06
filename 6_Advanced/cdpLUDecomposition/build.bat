rem nvcc -arch=sm_35 -rdc=true cdp_lu.cu cdp_lu_main.cu dgetrf.cu dgetf2.cu dlaswp.cu -lcublas_device -lcublas -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\libx64" -lcudadevrt -o cdpLU
