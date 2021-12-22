include("./shared.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D_LF_xpu_mpi.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D.jl")

# run the model for dam break in y-direction
H, xc = shallow_water_2D_xpu_mpi(; nx = n, ny = n, dam1D_x = false)
indsx = Int.(ceil.(LinRange(1, length(xc), 12)))
indsy = Int.(ceil.(LinRange(1, length(xc), 12)))
d = Dict(:X => xc[indsx], :H => H[indsx, indsy])

@testset "Ref-test shallow water 2D with dam break in y-direction" begin
      @test_reference "reftest-files/shallow_water_2D_xpu_mpi_mpi_y.bson" d by = comp
end

@reset_parallel_stencil()
