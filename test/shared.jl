using Test, ReferenceTests, BSON
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

n = 32 # grid size

# Reference test using ReferenceTests.jl
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2) for (v1,v2) in zip(values(d1), values(d2))])