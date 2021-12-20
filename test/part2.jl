include("../scripts-part2/2D_SWE/shallow_water_2D_LF_xpu_mpi.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D.jl")

using Test
@testset "Height H between baseline and MPI implementation for dam breaks in y-direction" begin
    H_xpu, xc = shallow_water_2D_xpu(; nx = 128, ny = 64, dam1D_x = false, do_visu = false)
    H_cpu = shallow_water_2D(; nx = 128, ny = 64, dam_x = false, do_visu = false)
    @test H_xpu ≈ H_cpu
end;

@testset "Height H between baseline and MPI implementation for dam breaks in x-direction" begin
    H_xpu, xc = shallow_water_2D_xpu(; nx = 128, ny = 64, dam1D_x = true, do_visu = false)
    H_cpu = shallow_water_2D(; nx = 128, ny = 64, dam_x = true, do_visu = false)
    @test H_xpu ≈ H_cpu
end;


## Reference Tests with ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-6) for (v1,v2) in zip(values(d1), values(d2))])

# run the model for dam break in x-direction
H_x, xc_x = shallow_water_2D_xpu(; nx = 64, ny = 32, do_viz = false)
# run the model for dam break in y-direction
H_y, xc_y = shallow_water_2D_xpu(; nx = 64, ny = 32, dam_x = false, do_viz = false)

inds = Int.(ceil.(LinRange(1, length(xc), 12)))
d1 = Dict(:X=>xc_x[inds], :H=>H_x[inds, inds, 15])
d2 = Dict(:X=>xc_y[inds], :H=>H_y[inds, inds, 15])

@testset "Ref-tests for 2D xpu perf" begin
	# @test_reference "reftest-files/shallow_water_2D_xpu_perf_x.bson" d1 by=comp
    	# @test_reference "reftest-files/shallow_water_2D_xpu_perf_y.bson" d2 by=comp
end