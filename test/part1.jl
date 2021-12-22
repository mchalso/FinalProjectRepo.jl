include("../scripts-part1/part1.jl")

H, xc_g = diffusion_3D(; do_visu=false)

# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(xc_g), 12)))
d = Dict(:X=> xc_g[inds], :H=>H[inds, inds, 15])

@testset "Ref-test diffusion 3D dual-time" begin
    @test_reference "reftest-files/test_1.bson" d by=comp
end
