# Part 2: another PDE

nx = 8 * 2 .^ (1:8)

precision = [2.968512, 2.980016, 2.989295, 2.987390, 3.029586, 3.216812, 2.854171, 3.390651]

opts = (ylims=(0, 4), linewidth=3, markersize=4, markershape=:circle, legend=false)
plot(nx, precision, framestyle=:box, xlabel="#grid points in each direction (nx=ny)", ylabel="H[nx/2, ny/2] [m]"; opts...)
savefig("plots/part-2/precision_scaling_2D")