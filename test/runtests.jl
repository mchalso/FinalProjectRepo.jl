using FinalProjectRepo
using Test, ReferenceTests, BSON

printstyled("Testing Part 1 of FinalProjectRepo.jl\n"; bold = true, color = :white)
include("part1.jl")

# Run tests for part 2 in save testing environment and the tests are split into different files because of MPI initialisation.
function runtests()
      exename = joinpath(Sys.BINDIR, Base.julia_exename())
      testdir = pwd()
      istestpart2(f) = endswith(f, ".jl") && startswith(basename(f), "part2_")
      testfiles = sort(filter(istestpart2, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

      nfail = 0
      printstyled("Testing Part 2 of FinalProjectRepo.jl\n"; bold = true, color = :white)

      for f in testfiles
            try
			run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
            catch ex
                  nfail += 1
            end
      end
      return nfail
end

exit(runtests())