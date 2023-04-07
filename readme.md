Note for anyone wishing to compile separately from the Visual Studio solution: 

The main files make use of c++17 features. Since NVCC as of now only supports up to C++14, the main files must be compiled with the system's main compiler.