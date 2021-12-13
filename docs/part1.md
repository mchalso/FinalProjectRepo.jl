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
![equation-pseudo](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%5Ctau%7D%20%3D%20-%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%20t%7D%20%2B%20D%20%5Cnabla%20%5E2%20C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) 
- For each 'real' timestep, we loop with respect to the 'virtual' time until the
equation's residual (`R_H`) is below a chosen threshold:

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

|                                         ![diffusion-3d-video](../plots/part-1/diffusion.mp4)                                        |
|:-----------------------------------------------------------------------------------------------------------------------------------:|
| Animation of the 3D steady-state diffusion being solved with the dual-time method. Each frame corresponds to one physical time-step |

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
| 1              | 256x256x256 | 4.19              | 83.604   |
| 2              |             |                   |          |
| 4              |             |                   |          |

#### Work-precision diagrams
Provide a figure depicting convergence upon grid refinement; report the
evolution of a value from the quantity you are diffusing for a specific location
in the domain as function of numerical grid resolution. Potentially compare
against analytical solution.

Provide a figure reporting on the solution behaviour as function of the solver's
tolerance. Report the relative error versus a well-converged problem for various
tolerance-levels. 

## Discussion
Discuss and conclude on your results

<!-- ## References -->

[^1]: [https://gadgetversus.com/processor/amd-ryzen-5-3600-specs/](https://gadgetversus.com/processor/amd-ryzen-5-3600-specs/) 
[^2]: [https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632](https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632) 
