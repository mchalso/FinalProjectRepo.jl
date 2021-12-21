# Part 2: [your personal project]
Solving PDEs of your choice using Julia on (multi-) GPUs -or- applying some performance optimisation to an exisiting or "bake-off" problem.

## Intro
What's all about. Brief overview about:
- the process
- the equations
- the aims
- ...

## Methods
The methods to be used:
- spatial and temporal discretisation
- solution approach
- hardware
- ...

## Results

In this section, we discuss the results obtained for our implementation.

### 2D shallow water

![2D SWE with 2D dam break](../plots/part-2/shallow_water_2D_xpu_2D-dam.gif)

| Animation of the 2D shallow water equations being solved with a 2D dam break. Each frame corresponds to one physical time-step. |
|:--:|

![2D SWE 1D x-dir dam break](../plots/part-2/shallow_water_2D_xpu_1D-dam-x.gif)

| Animation of the 2D shallow water equations being solved with a 1D dam break in x-direction. Each frame corresponds to one physical time-step. |
|:--:|

![2D SWE 1D y-dir dam break](../plots/part-2/shallow_water_2D_xpu_1D-dam-y.gif)

| Animation of the 2D shallow water equations being solved with a 1D dam break in y-direction. Each frame corresponds to one physical time-step. |
|:--:|

TODO: comment on visualization

### Performance

For the performance measurements we use the 2D xpu with MPI solver of the shallow water equations.

#### Memory throughput

Strong-scaling on CPU and GPU -> optimal "local" problem sizes.

We test the performance of the algorithm using only a single thread in a machine
with a single Intel Core i7 running at 2.5 GHz.

| Domain size     | Throughput (GB/s) | Time (s) |
| --------------- | ---------------   | -------- |
| 32              | 0.75              | 0.654   |
| 64              | 0.96              | 2.037    |
| 128             | 0.96              | 8.204    |
| 256             | 1.01            | 31.062  |
| 512             | 0.979            | 128.367  |

The Intel Core i7 has a theoretical memory bandwidth of 25.6GB/s[^1]. This means
that we are bounded by our computations, which makes sense since single CPU
cores are not efficient at tackling these kind of highly parallel problems
because of their sequential nature. 

We also conduct the same experiment on the GTX TITAN X, yielding the following
results:

| Domain size     | Throughput (GB/s) | Time (s) |
| --------------- | ---------------   | -------- |
| 32x32              | 0.13             | 3.727  |
| 64x64              | 0.50             | 3.921  |
| 128x128             | 1.61             | 4.870  |
| 256x256             | 3.66             | 8.586  |
| 512x512             | 7.79             | 16.129   |
| 1024x1024		| 13.60		 | 36.862	|
| 2048x2048		| 15.40		 | 130.262	|
| 4096x4096		| 17.10		 | 470.429 |
| 8192x8192		| 17.10		 | 1882.127|

The TITAN X has a theoretical maximum memory bandwidth of 336.6GB/s[^2]. Since our results are an order of magnitude lower than this theoretical maximum, it is clear that our performance is also compute bounded. This makes sense since our algorithm has some parts which are not optimally paralellisable (lot of control statements inside kernel) and since this model of GPU is fairly outdated, and back in the day the gap between memory and compute performances was not as big.

#### Weak scaling

Having found the optimal local problem size for the GTX TITAN X in the previous
section (4096), we now run the same problem in 1, 2 and 4 GPUs in order to
compare the total throughput achieved.

| Number of GPUs | Domain size | Throughput (GB/s) | Time (s) |
| -----------    | ----------  | ----------------  | -------- |
| 1              | 4096x4096 	| 17.10              | 470.429    |
| 2              | 8190x4096 	| 24.20             | 333.018    |
| 4              | 8190x8190 	| 24.00             | 334.780    |

In order to get these results, we used the command:

```bash
~/.julia/bin/mpiexecjl -n nproc julia --project scripts-part2/2D_SWE/shallow_water_2D_LF_xpu_mpi.jl
```
The script is called with (n is the grid size):
```julia
shallow_water_2D_xpu(; nx = n, ny = n, dam2D = false, dam1D_x = true, do_visu = false)
```

#### Work-precision diagrams

We now perform an evaluation of the algorithm convergence given a certain grid
refinement. Ideally, the more density of points we have the more our results
will imitate reality. To validate this, we are going to check the value at the
center of the domain given different grid sizes. We expect to observe that the
value converges as we increase the grid size.

We can see our results in the following table. 

| Size | H[nx/2, ny/2] m |
|------|------------------|
| 16   | 2.968512          |
| 32   | 2.980016          |
| 64   | 2.989295          |
| 128  | 2.987390          |
| 256  | 3.029586          |
| 512  | 3.216812          |
| 1024 | 2.854171		|
| 2048 | 3.390651		|

And the following plot shows it graphically.

![work precission](../plots/part-2/precision_scaling_2D.png) 

## Discussion



TODO: Why work precision scaling doesn't converge.

## References
[^1]: [https://www.intel.com/content/www/us/en/products/sku/83504/intel-core-i74870hq-processor-6m-cache-up-to-3-70-ghz/specifications.html](https://www.intel.com/content/www/us/en/products/sku/83504/intel-core-i74870hq-processor-6m-cache-up-to-3-70-ghz/specifications.html) 
[^2]: [https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632](https://www.techpowerup.com/gpu-specs/geforce-gtx-titan-x.c2632) 