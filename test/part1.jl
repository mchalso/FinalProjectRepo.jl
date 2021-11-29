# Testing part 1

include("../scripts-part1/part1.jl") # modify to include the correct script

# Add unit and reference tests


# Reference test using ReferenceTests.jl
"Compare all dict entries"
comp(d1, d2) = keys(d1) == keys(d2) && all([ isapprox(v1, v2; atol = 1e-5) for (v1,v2) in zip(values(d1), values(d2))])
inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
d = Dict(:X=> Xc_g[inds], :H_g=>H_g[inds])

@testset "Ref-file" begin
    @test_reference "reftest-files/test.bson" d by=comp
end
