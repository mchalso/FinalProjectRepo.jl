
# Part 2: 2D Shallow Water Equations Solver
Solution of the shallow water equations in one and two dimensions. The equations are solved using Julia on multi-GPUs.

## Intro
The shallow water equations are a set of hyperbolic partial differential equations (PDEs) for describing shallow fluid flows. The shallow water equations are derived from depth-integration of the Navier-Stokes Equations, which describe continuity of mass and continuity of momentum of fluids. The shallow water equations assume a hydrostatic pressure distribution and constant velocity throughout fluid depth, along with the condition that horizontal length scale is significantly larger than than the vertical scale. The 2D shallow water equations include equations for conservation of mass, conservation of momentum in the x-direction, and conservation of momentum in the y-direction. The equations are shown here in conservative form for a horizontal bed, neglecting friction and viscous forces:

### Conservation of mass
![equation-2DSWE-continuity](http://www.sciweavers.org/download/Tex2Img_1640034852.jpg)

### Conservation of momentum in X-direction
![equation-2DSWE-momentumx](http://www.sciweavers.org/download/Tex2Img_1640035257.jpg)

### Conservation of momentum in Y-direction  
![equation-2DSWE-momentumy](http://www.sciweavers.org/download/Tex2Img_1640035339.jpg)

The purpose of this shallow water equations solver is to model an instantaneous dam breach. The model domain has a length of 40 meters and a width of 20 meters. Half of the domain (a 20 meter x 20 meter region) has an initial water level of 5 meters, while the other half (also a 20 meter x 20 meter region) has an initial water level of 1 meter. The model setup matches that of a test case used in the validation of the software BASEMENT version 2.8, Test Case H_1 "Dam break in a closed channel" [^1]

<!-- What's all about. Brief overview about: -->
<!-- - the process -->
<!-- - the equations -->
<!-- - the aims -->
<!-- - ... -->

## Methods

### Spatial discretization
The model domain was spatially discretized into square-shaped (dx = dy) elements, with edge lengths of approximately 0.08 meters. There are 512 elements in the x-direction, and 256 elements in the y-direction. The model was initially tested with smaller sized elements, and then elements were coarsened to the maximum size at which accuracy was acceptable, in order to reduce computational cost.

### Temporal discretization
The temporal discretization (dt) is adapated throughout the simulation based on the CFL condition. The timestep must be small enough so that information cannot travel greater than the distance between computational elements (dx), with additional reduction to ensure convergence in 2D modeling. The dt is adjusted at the beginning of each time step, based on the maximum velocity in the system. The calculation of dt is shown here:  

![equation-dt](http://www.sciweavers.org/download/Tex2Img_1640039918.jpg)

### Solution approach: Lax-Friedrichs Method
The 2D shallow water equations were solved by utilizing the Lax-Friedrichs Method [^2][^3]. The Lax-Friedrichs Method is a forward in time, centered in space numerical scheme. For an equation of the form  

![equation-LFM-basic](http://www.sciweavers.org/download/Tex2Img_1640042032.jpg)

f(u) and g(u) are flux functions of u, artificial dissipation is applied to the flux functions in order to mitigate numerical instabilities. "Corrected" flux functions F(u) and G(u) are solved for at cell edges:  

![equation-LFM-F](http://www.sciweavers.org/download/Tex2Img_1640043308.jpg)  

![equation-LFM-G](http://www.sciweavers.org/download/Tex2Img_1640043405.jpg)  

where λ<sub>x</sub> and λ<sub>y</sub> are signal speeds in the x and y directions. λ<sub>x</sub> and λ<sub>y</sub> are functions of the adapting dt, and are calculated as:

![equation-lambdax](http://www.sciweavers.org/download/Tex2Img_1640044755.jpg)

![equation-lambday](http://www.sciweavers.org/download/Tex2Img_1640044800.jpg)

The "corrected" flux functions F(u) and G(u) are then used to solved for the next iteration of the solution matrix U:  

![equation-LFM-U](http://www.sciweavers.org/download/Tex2Img_1640043968.jpg)  


<!--The methods to be used: -->
<!--- spatial and temporal discretisation -->
<!--- solution approach -->
- hardware
- ...

## Results

### The physics you are resolving

### Performance

#### Memory throughput

#### Weak scaling

#### Work-precision diagrams

## Discussion

## References
<!-- ## References -->
[^1]: [Vetsch, D., Siviglia, A., Caponi, F., Ehrbar, D., Gerke, E., Kammerer, S., Koch, A., Peter, S., Vanzo, D., Vonwiller, L., Facchini, M., Gerber, M., Volz, C., Farshi, D., Mueller, R., Rousselot, P., Veprek, R., & Faeh, R. (2018). System manuals of basement, Version 2.8., ETH Zurich: Zurich, Switzerland.](http://people.ee.ethz.ch/~basement/baseweb/download/documentation/BMdoc_Testcases_v2-8-1.pdf)
[^2]: [Lax, P.D. (1954), Weak solutions of nonlinear hyperbolic equations and their numerical computation. Comm. Pure Appl. Math., 7: 159-193.](https://doi.org/10.1002/cpa.3160070112)  
[^3]: [Rezzolla, L. (2020), Numerical Methods for the Solution of Partial Differential Equations. Institute for Theoretical Physics, Frankfurt, Germany.](https://itp.uni-frankfurt.de/~rezzolla/lecture_notes/2010/fd_evolution_pdes_lnotes.pdf)
