include("./shared.jl")
include("../scripts-part2/1D_SWE/shallow_water_1D_LF_xpu.jl")
include("../scripts-part2/1D_SWE/shallow_water_1D_LF.jl")

@testset "Height H between baseline and 1D xpu implementation" begin
      H_xpu = shallow_water_1D_xpu(; nx = n)
      H_cpu = shallow_water_1D(; nx = n)
      @test isapprox(H_xpu, H_cpu)
end;