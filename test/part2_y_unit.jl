include("./shared.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D_LF_xpu_mpi.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D_LF_xpu.jl")
include("../scripts-part2/2D_SWE/shallow_water_2D.jl")

@testset "Height H between baseline and xpu_mpi implementation of dam break in y-direction" begin
      H_mpi, xc = shallow_water_2D_xpu_mpi(; nx = n, ny = n, dam1D_x = false)
	@reset_parallel_stencil()
	H_xpu = shallow_water_2D_xpu(; nx = n, ny = n, dam1D_x = false)
	@reset_parallel_stencil()
      H_cpu = shallow_water_2D(; nx = n, ny = n, dam_x = false)
      @test isapprox(H_xpu, H_cpu)
	@test isapprox(H_mpi, H_cpu)
end;