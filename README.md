
OPENMP COMPUTING OF A REFERENCE SOLUTION FOR COUPLED LORENZ SYSTEM ON [0,400]

To run the program Program_CLorenz.c,  GMP library (GNU multiple precision library) is needed: https://gmplib.org/

The precision in the program is set at 7488 bits ~2254 decimal digits. The order of the method is 2580. The solution is computed on [0,400].  At every 10 time units the solution is printed with 60 digits in the file “res_400.txt”. We took as initial conditions (x(0);y(0); z(0)) = (5;5;10); (X(0);Y(0);Z(0)) = (5;5;10),  the same as those in [1], where the solution was computed up to 100. We repeated the benchmark table from [1].

[1] Wang, Pengfei, et al. "Clean numerical simulation for some chaotic systems using the   parallel   multiple-precision Taylor scheme." Chinese science bulletin 59.33 (2014): 4465-4472.

The main points of the OpenMP parallelization are:
1) For given i make an explicit parallel reduction for the sums in formulas (2). 
2) After computation of sums for given i, compute each formula for the 6 components   xi+1, yi+1, zi+1, Xi+1, Yi+1, Zi+1  independently in parallel.  
3) After computation of all the derivatives up to the N-th, compute the variable step-size in a single section. 
4) Use Horner’s rule for evaluation of all 6 components of the solution independently in parallel

OpenMP  parallel technology has its own importance for multiple precision Taylor Series Method, because: 
1) OpenMP is simpler than MPI since the communication between threads is realized by shared memory and we do not need to learn special libraries for packaging and unpackaging of multiple precision numbers.
2) OpenMP is slightly faster than pure MPI, most likely because the additional overhead for packaging and unpackaging of the MPI massages. 
3) OpenMP uses less memory, since the algorithm does not allow domain decomposition and the computational domain has to be multiplied by the number of MPI processes. 

Comment: Using SPMD programming pattern we can divide the threads in groups and make each group to compute equal number of sums. In this case we will have a little performance benefit, because for the small values of the index i the unused threads will be less and also the difference from the perfect load balance between threads will be less. However, this approach is not general, because it strongly depends on the number of sums for reduction and the number of available threads. The approach is not applicable in our case of Coupled Lorenz system, because the number of sums is 5 and the threads are 32. The approach is applicable for example for the classical Lorenz system, where the number of sums are 2.
