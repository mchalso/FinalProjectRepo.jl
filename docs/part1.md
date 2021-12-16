# Part 1: 3D multi-XPUs diffusion solver

Steady state solution of a diffusive process for given physical time steps using
the pseudo-transient acceleration (using the so-called "dual-time" method).

## Intro

This part of the project consists in solving the 3D diffusion equation:

![equation-diffusion](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%20t%7D%20%3D%20D%20%5Cnabla%20%5E2%20C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

To solve it in an efficient manner, we'll make use of the Julia libraries
`ParallelStencil` and `ImplicitGlobalGrid`, which make the process of writing
high performant multi-xpu applications almost trivial.

Moreover, we'll make use of advanced rendering capabilities of the library
`Makie` to plot the entirety of the 3D domain.

<!-- What's all about. Brief overview about: -->
<!-- - the process -->
<!-- - the equations -->
<!-- - the aims -->
<!-- - ... -->

## Methods

We are interested in implementing an iterative algorithm to find the
steady-state of a diffusive process. In order to do so and with the goal of
increasing the computational efficiency, we are going to make use of the method
known as dual-time. 

The dual-time method is a type of implicit solution that makes use of two
derivatives in pseudo-time, improving the convergence of the steady-state
solution thus reducing the number of iterations needed. We proceed to solve this
problem as explained in lecture 4:

- We start with the 3D diffusive process:

![equation-diffusion](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%20t%7D%20%3D%20D%20%5Cnabla%20%5E2%20C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

- We then move every element to the right and introduce the pseudo-time
integration:

![equation-pseudo](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%5Ctau%7D%20%3D%20-%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%20t%7D%20%2B%20D%20%5Cnabla%5E2%20H&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

- For each 'real' timestep, we loop with respect to the 'virtual' time until the
equation's residual (`R_H`) is below a chosen threshold.

Regarding the spatial discretisation, we will make a cubic grid with equidistant
points in every direction. This scheme will simplify the coding of the
algorithms and the storage of the domain in the system's memory.

Since we are making use of `ParallelStencil` and `ImplicitGlobalGrid`, our code
runs on both CPU, GPU and multi-CPU/GPU systems without the need of making any
change. We will test our code in a single CPU (Ryzen 3600 @ 3.6 GHz) as well as
the Octopus server (4x NVIDIA TITAN X).

<!-- The methods to be used: -->
<!-- - spatial and temporal discretisation -->
<!-- - solution approach -->
<!-- - hardware -->
<!-- - ... -->

## Results

In this section, we discuss the results obtained for our implementation.

### 3D diffusion

https://user-images.githubusercontent.com/8024691/145982014-db650778-876d-4e5c-a6a1-6e2bc7e4412d.mp4

| Animation of the 3D steady-state diffusion being solved with the dual-time method. Each frame corresponds to one physical time-step. |
|:--:|

In this animation we can see three different views of the diffusion process. On
the top one, a 3D representation of the whole domain is shown, rotating between
frames so it is easier to appreciate its depth. On the bottom left, we see a
slice of the XY plane at `z=16`. On the bottom right, we see another slice, but
this time of the YZ plane along `x=16`. We observe that due to the symmetry of
the initialization, these two planes look exactly the same.

<!-- Report an animation of the 3D solution here and provide and concise description -->
<!-- of the results. _Unleash your creativity to enhance the visual output._ -->

### Performance
<!-- Briefly elaborate on performance measurement and assess whether you are compute -->
<!-- or memory bound for the given physics on the targeted hardware. -->

#### Memory throughput
Strong-scaling on CPU and GPU -> optimal "local" problem sizes.

We test the performance of the algorithm using only a single thread in a machine
with a single Ryzen 3600 running at 3.6 GHz.

| Domain size     | Throughput (GB/s) | Time (s) |
| --------------- | ---------------   | -------- |
| 32              | 5.00              | 0.2912   |
| 64              | 4.67              | 4.870    |
| 128             | 4.39              | 84.56    |
| 256             | 4.19              | 1471.75  |

The Ryzen 3600 has a theoretical memory bandwidth of 47,68GB/s[^1]. This means
that we are bounded by our computations, which makes sense since single CPU
cores are not efficient at tackling these kind of highly parallel problems
because of their sequential nature. Since it is cheaper to move memory around
than to compute, it is logical that the smaller the domain, the less
computations we have to do and the more throughput we have.

We also conduct the same experiment on the GTX TITAN X, yielding the following
results:

| Domain size     | Throughput (GB/s) | Time (s) |
| --------------- | ---------------   | -------- |
| 32              | 15.70             | 0.09245  |
| 64              | 39.00             | 0.58145  |
| 128             | 64.60             | 5.75459  |
| 256             | 73.70             | 83.8132  |
| 512             | 73.30             | 1406.2   |

The TITAN X has a theoretical maximum memory bandwidth of 336.6GB/s[^2]. Since
we are so far behind this theoretical maximum, it is clear that our performance
is also compute bounded. This makes sense since this model of GPU is fairly
outdated, and back in the day the gap between memory and compute performances
was not as big.

#### Weak scaling

Having found the optimal local problem size for the GTX TITAN X in the previous
section (256), we now run the same problem in 1, 2 and 4 GPUs in order to
compare the total throughput achieved.

| Number of GPUs | Domain size | Throughput (GB/s) | Time (s) |
| -----------    | ----------  | ----------------  | -------- |
| 1              | 256x256x256 | 74.6              | 16.20    |
| 2              | 510x256x256 | 141.0             | 31.78    |
| 4              | 510x510x256 | 259.0             | 37.20    |

In order to get these results, we used the command:

```bash
~/.julia/bin/mpiexecjl -n nproc julia --project scripts-part1/part1.jl
```

#### Work-precision diagrams

We now perform an evaluation of the algorithm convergence given a certain grid
refinement. Ideally, the more density of points we have the more our results
will imitate reality. To prove this, we are going to check the value of the
center of the domain given different grid sizes. We expect to observe that the
value converges as we increase the grid size.

We can see our results in the following table.

| Size | H[n/2, n/2, n/2] |
|------|------------------|
| 16   | 0.22058          |
| 32   | 0.22548          |
| 64   | 0.22659          |
| 128  | 0.22686          |
| 256  | 0.22693          |
| 512  | 0.22696          |

And the following plot shows it graphically.

![work precission](https://github.com/mchalso/FinalProjectRepo.jl/blob/main/plots/part-1/work-precission-diagram.png?raw=true) 

Now we are going to test how the solution changes by adjusting the tolerance of
the solver. We are going to consider the problem where `nx=ny=nz=32` and
`tol=1e-8` as a baseline. To compute the errors, we are going to use the norm of
the difference of the new `H` and the baseline `H`. We get the following values:

| Tol  | Error      |
|------|------------|
| 1e-8 | 0          |
| 1e-7 | 4.33232e-6 |
| 1e-6 | 3.11549e-5 |
| 1e-5 | 6.39017e-4 |
| 1e-4 | 7.43127e-3 |
| 1e-3 | 5.65919e-2 |
| 1e-2 | 0.393818   |
| 1e-1 | 0.625191   |
| 1e-0 | 0.625191   |

That have the corresponding plot:

![tolerance-error](https://github.com/mchalso/FinalProjectRepo.jl/blob/main/plots/part-1/tolerance-error-diagram.png?raw=true) 

We can see that from `1e-1` onward, the inner loop is ended by the expression
`iter < maxIter` and thus we get the same error.

## Discussion

In this part of the final project, we have implemented a multi-xpu 3D diffusion
solver, that is, without changing the implementation we are able to run this
solver in either one or multiple machines, using GPU or CPU. We have tested our
implementation against the reference test given by the teaching staff, and
evaluated its performance on the CPU of an average desktop (Ryzen 3600) as well
as at the Octopus supercomputer (with up to 4 GTX TITAN X). In both cases, we
reached at the conclusion that we were compute bounded. This makes sense because
the GPUs used are not recent models, were the gap between the compute and memory
speeds is much more significant.

To test the performance of the multi-gpu application, first we perform tests of
strong scaling in order to obtain the optimum problem size for our hardware.
Once we know it, we proceed to execute the exact same problem on multiple
processors making use of `ImplicitGlobalGrid` (weak scaling). We can see that
our throughput increases with the number of GPUs used, but we can also
appreciate that the increase is not linear and slows down with the number of
GPUs used. This is because of the need to synchronize the global problem
updating the halo at each timestep.

We conclude this part by making two plots: the work-precission diagram and the
tolerance-error diagram. The latter shows how the computed steady-state diverges
from a well-converged problem the more we decrease the tolerance, we can see
that this growth is linear (since the x-scale is logarithmic). Whereas for the
work-precission diagram, we show how the value of the exact middle of the domain
evolves when we change the domain resolution. As expected, it converges
to its true value the more resolution we use.

<!-- ## References -->

[^1]: [https://gadgetversus.com/processor/amd-ryzen-5-3600-specs/](https://gadgetversus.com/processor/amd-ryzen-5-3600-specs/) 
[^2]: [https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632](https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632) 
