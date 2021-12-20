# Part 2: 2D Shallow Water Equations Solver
Solution of the shallow water equations in one and two dimensions. The equations are solved using Julia on multi-GPUs.

## Intro
The shallow water equations are a set of hyperbolic partial differential equations (PDEs) for describing shallow fluid flows. The shallow water equations are derived from depth-integration of the Navier-Stokes Equations, which describe continuity of mass and continuity of momentum of fluids. The shallow water equations assume a hydrostatic pressure distribution and constant velocity throughout fluid depth, along with the condition that horizontal length scale is significantly larger than than the vertical scale. The 2D shallow water equations are shown here in conservative form:

md"""
$$ \frac{∂h}{∂t} + \frac{∂(hu)}{∂x} + \frac{∂(hv)}{∂y} = 0,$$  
\frac{∂(hu^2 + 0.5gh^2)}{∂t} = c^2 ∇^2 u~,$$
$$ \frac{∂^2u}{∂t^2} = c^2 ∇^2 u~,$$

"""

<!-- What's all about. Brief overview about: -->
<!-- - the process -->
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

### The physics you are resolving

### Performance

#### Memory throughput

#### Weak scaling

#### Work-precision diagrams

## Discussion

## References
