# Final Project Repo

[![Build Status](https://github.com/mchalso/FinalProjectRepo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mchalso/FinalProjectRepo.jl/actions/workflows/CI.yml?query=branch%3Amain)

[👆 Make sure to include unit and reference tests. We provide you with a [reference test](/test/part1.jl) (using [ReferenceTests.jl](https://github.com/JuliaTesting/ReferenceTests.jl)) you can benchmark your implementation against. Adapt it to your needs and ensure your 3D diffusion solver to return `H_g` and `Xc_g`, the global solution array and the global x-coord vector, respectively.]

[Add some short info here about the project, an abstract in a sense, and link to the documentation for [**Part-1**](/docs/part1.md) and [**Part-2**](/docs/part2.md).]

## Meta-Info (delete this)

This project was generated with
```julia
using PkgTemplates
Template(;dir=".",
          plugins=[
                   Git(; ssh=true),
                   GitHubActions(; x86=true)],
        )("FinalProjectRepo")
```
Additionally, to the files generated by `PkgTemplates`, the following files and folders were added
- `scripts-part1/` and `scripts-part2/` which should contain the scripts for part 1 (solving the diffusion equation) and part 2 (solving your own equation)
- `docs/` the documentation (aka your final report), one for each part
- `test/part*.jl` testing scripts

Adapting this to your needs would entail:
- copy this repository (or clone)
- adapt the files (don't forget the LICENSE)
