
# Part 2: 2D Shallow Water Equations Solver
Solution of the shallow water equations in one and two dimensions. The equations are solved using Julia on multi-GPUs.

## Intro
The shallow water equations are a set of hyperbolic partial differential equations (PDEs) for describing shallow fluid flows. The shallow water equations are derived from depth-integration of the Navier-Stokes Equations, which describe continuity of mass and continuity of momentum of fluids. The shallow water equations assume a hydrostatic pressure distribution and constant velocity throughout fluid depth, along with the condition that horizontal length scale is significantly larger than than the vertical scale. The 2D shallow water equations include equations for conservation of mass, conservation of momentum in the x-direction, and conservation of momentum in the y-direction. The equatoins are shown here in conservative form for a horizontal bed, neglecting friction and viscous forces.

### Conservation of Mass
![equation-2DSWE-continuity](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%20%5Cdelta%20h%7D%7B%20%5Cdelta%20t%7D%20%2B%20%5Cfrac%7B%20%5Cdelta%20hu%7D%7B%20%5Cdelta%20x%7D%20%2B%20%5Cfrac%7B%20%5Cdelta%20hv%7D%7B%20%5Cdelta%20y%7D%20%3D%200&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)


### Conservation of Momentum in X-direction
![equation-2DSWE-momentumx](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%20%5Cdelta%20hu%7D%7B%20%5Cdelta%20t%7D%20%2B%20%5Cfrac%7B%20%5Cdelta%20%28hu%5E2%20%2B0.5gh%5E2%29%7D%7B%20%5Cdelta%20x%7D%20%2B%20%5Cfrac%7B%20%5Cdelta%20huv%7D%7B%20%5Cdelta%20y%7D%20%3D%200&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)


### Conservation of Momentum in Y-direction
![equation-2DSWE-momentumy](http://www.sciweavers.org/tex2img.php?eq=\frac{%20\delta%20hu}{%20\delta%20t}%2B\frac{%20\delta%20huv}{%20\delta%20x}%2B\frac{%20\delta%20(hv^2%20%2B0.5gh^2)}{%20\delta%20y}%20%3D%200&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)


The purpose of this shallow water equations solver is to model an instantaneous dam breach. The model domain has a length of 40 meters and a width of 20 meters. Half of the domain (a 20 meter x 20 meter region) has an initial water level of 5 meters, while the other half (also a 20 meter x 20 meter region) has an initial water level of 1 meter. The model setup matches that of a test case used in the validation of the software BASEMENT version 2.8, Test Case H_1 "Dam break in a closed channel."

<!-- What's all about. Brief overview about: -->
<!-- - the process -->
<!-- - the equations -->
<!-- - the aims -->
<!-- - ... -->

## Methods
The methods to be used:
- spatial and temporal discretisation
- solution approach
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
