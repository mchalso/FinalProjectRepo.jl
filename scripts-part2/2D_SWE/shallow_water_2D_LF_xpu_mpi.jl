const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, ImplicitGlobalGrid
import MPI

macro H_dx(ix, iy)
    esc(:(H[$ix+1, $iy] - H[$ix, $iy]))
end
macro H_dy(ix, iy)
    esc(:(H[$ix, $iy+1] - H[$ix, $iy]))
end
macro Hv_dy(ix, iy)
    esc(:(H[$ix, $iy+1] * v[$ix, $iy+1] - H[$ix, $iy] * v[$ix, $iy]))
end
macro Hv_py(ix, iy)
    esc(:(H[$ix, $iy+1] * v[$ix, $iy+1] + H[$ix, $iy] * v[$ix, $iy]))
end
macro Huv_px(ix, iy)
    esc(:(H[$ix+1, $iy] * u[$ix+1, $iy] * v[$ix+1, $iy] + H[$ix, $iy] * u[$ix, $iy] * v[$ix, $iy]))
end
macro Huv_dx(ix, iy)
    esc(:(H[$ix, $iy+1] * v[$ix, $iy+1] * u[$ix, $iy+1] - H[$ix, $iy] * v[$ix, $iy] * u[$ix, $iy]))
end

"""
	compute_H!(H_new, H, G, u, v, dt_dx, dt_dy, size_H1_2, λx_2, λy_2)

ParallelStencil kernel that computes a time step of the shallow water 2D
process.
"""
@parallel_indices (ix, iy) function compute_H!(
    H_new,
    H,
    u,
    v,
    dt_dx,
    dt_dy,
    size_H1_2,
    λx_2,
    λy_2,
)
    if (ix <= size_H1_2 && iy <= size(H, 2))
        temp = H[ix+1, iy] -
               dt_dx * (
            1 / 2 * (H[ix+2, iy] * u[ix+2, iy] - H[ix, iy] * u[ix, iy]) -
            λx_2 * (@H_dx(ix + 1, iy) - @H_dx(ix, iy))
        )
        if (iy == size(H, 2))
            H_new[ix+1, iy] = (
                temp + dt_dy * (1 / 2 * @Hv_py(ix + 1, iy - 1) - λy_2 * @H_dy(ix + 1, iy - 1))
            )
        elseif (iy == 1)
            H_new[ix+1, iy] = (
                temp - dt_dy * (1 / 2 * @Hv_py(ix + 1, iy) - λy_2 * @H_dy(ix + 1, iy))
            )
        else
            H_new[ix+1, iy] = (
                temp - dt_dy * (1 / 2 * (@Hv_dy(ix + 1, iy) + @Hv_dy(ix + 1, iy - 1)) - λy_2 * (@H_dy(ix + 1, iy) - @H_dy(ix + 1, iy - 1)))
            )
        end
    end
    return
end

"""
	compute_u!(u_new, u, H_new, H, dt_dx, dt_dy, size_H1_2, g) 	

ParallelStencil kernel that computes the Lax-Friedrichs for the momentum in x-direction of the shallow water
equations.
"""
@parallel_indices (ix, iy) function compute_u!(
    u_new,
    u,
    H_new,
    H,
    dt_dx,
    dt_dy,
    size_H1_2,
    g,
)
    if (ix <= size_H1_2 && iy <= size(H, 2))
        u_new[ix+1, iy] =
            (
                1 / 2 * (H[ix, iy] * u[ix, iy] + H[ix+2, iy] * u[ix+2, iy]) -
                dt_dx * 1 / 2 * (
                    H[ix+2, iy] * u[ix+2, iy]^2 + 0.5 * g * H[ix+2, iy]^2 -
                    H[ix, iy] * u[ix, iy]^2 - 0.5 * g * H[ix, iy]^2
                )
            ) / H_new[ix+1, iy]
    end
    return
end

"""
	compute_v!(v_new, v, H_new, H, u, dt_dx, dt_dy, size_H2_2, g)	

ParallelStencil kernel that computes the Lax-Friedrichs for the momentum in y-direction of the shallow water
equations.
"""
@parallel_indices (ix, iy) function compute_v!(
    v_new,
    v,
    H_new,
    H,
    u,
    dt_dx,
    dt_dy,
    size_H2_2,
    g,
)
    if (ix <= size(H, 1) && iy <= size_H2_2)
        temp = 1 / 2 * (H[ix, iy] * v[ix, iy] + H[ix, iy+2] * v[ix, iy+2]) - dt_dy * 1 / 2 * (
            H[ix, iy+2] * v[ix, iy+2]^2 - H[ix, iy] * v[ix, iy]^2 +
            0.5 * g * H[ix, iy+2]^2 - 0.5 * g * H[ix, iy]^2
        )
        if (ix == size(H, 1))
            v_new[ix, iy+1] =
                (
                    temp + dt_dx * 1 / 2 * @Huv_px(ix - 1, iy + 1)
                ) / H_new[ix, iy+1]
        elseif (ix == 1)
            v_new[ix, iy+1] =
                (
                    temp - dt_dx * 1 / 2 * @Huv_px(ix, iy + 1)
                ) / H_new[ix, iy+1]
        else
            v_new[ix, iy+1] =
                (
                    temp - dt_dx * 1 / 2 * (@Huv_dx(ix + 1, iy + 1) + @Huv_dx(ix - 1, iy + 1))
                ) / H_new[ix, iy+1]

        end
    end
    return
end

"""
	bc_x!(A)

Copy boundary conditions in x dimension of array `A`.
"""
@parallel_indices iy function bc_x!(A::Data.Array)
    A[1, iy] = A[2, iy]
    A[end, iy] = A[end-1, iy]
    return
end

"""
	shallow_water_2D_xpu(; nx, ny, dam1D_x, do_visu)

2D shallow water equations solver for an instantaneous dam break.
The Lax-Friedrichs Method was applied to the continuity equation.
Geometry (length of 40 meters, width of 20 meters) and initial conditions 
(half of domain has initial water level of 5 meters, other half is dry) 
matches BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel."
Solution of momentum equations requires division by H. Therefore a minimum
H must be set throughout the domain, to avoid numerical instability.
As parameters, we can modify:
    - `nx`: number of discretised cells for x dimension.
    - `ny`: number of discretised cells for y dimension.
    - `dam1D_x`: if true, 1D dam break in x-direction, else 1D dam break in y-direction. dam2D needs to be false to activate 1D dam break. 
    - `dam2D`: if true, 2D dam break
    - `do_visu`: if true, each physical time step will be ploted.
"""
@views function shallow_water_2D_xpu(;
    # Numerics
    nx = 512,
    ny = 256,
    # Modeling
    dam1D_x = true,
    dam2D = false,
    # Visualisation
    do_visu = false
)
    # Physics
    Lx, Ly = 40.0, 20.0
    g = 9.81
    u_max = 5 #from review of results with very small timesteps
    H_init = 5.0
    # Numerics
    nout = 100
    H_min = 1.0
    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1)  # Initialization of MPI and more...
    @static if USE_GPU
        select_device()
    end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy = Lx / nx_g(), Ly / ny_g()
    nt = 10000
    xc, yc = LinRange(dx / 2, Lx - dx / 2, nx), LinRange(dy / 2, Ly - dy / 2, ny)
    # Array initialisation
    H = @zeros(nx, ny) .+ H_min
    if (dam1D_x && !dam2D)
        H[1:round(Int64(nx / 2)), :] = @zeros(round(Int64(nx / 2)), ny) .+ H_init #1D dam break in x-direction
    elseif (!dam1D_x && !dam2D)
        H[:, 1:round(Int64(ny / 2))] = zeros(Float64, nx, round(Int64(ny / 2))) .+ H_init #1D dam break in y direction
    else
        H[1:round(Int64(nx/2)), 1:round(Int64(ny/2))] = zeros(Float64, round(Int64(nx/2)), round(Int64(ny/2))) .+ H_init #2D dam break
    end
    H_new = copy(H)
    u = @zeros(nx, ny)
    u_new = copy(u)
    v = @zeros(nx, ny)
    v_new = copy(v)
    size_H1_2, size_H2_2 = size(H, 1) - 2, size(H, 2) - 2

    # Preparation of visualisation
    if do_visu
        if (me == 0)
            ENV["GKSwstype"] = "nul"
            if isdir("plots/part-2/temp") == false
                mkdir("plots/part-2/temp")
            end
            loadpath = "./plots/part-2/temp"
            anim = Animation(loadpath, String[])
            println("Animation directory: $(anim.dir)")
        end
        nx_v, ny_v = (nx - 2) * dims[1], (ny - 2) * dims[2]
        if (nx_v * ny_v * sizeof(Data.Number) > 0.8 * Sys.free_memory())
            error("Not enough memory for visualization.")
        end
        H_v = zeros(nx_v, ny_v) # global array for visu
        H_inn = zeros(nx - 2, ny - 2) # no halo local array for visu
        Xi_g, Yi_g = LinRange(dx + dx / 2, Lx - dx - dx / 2, nx_v), LinRange(dy + dy / 2, Ly - dy - dy / 2, ny_v) # inner points only
    end
    t_tic = 0.0
    niter = 0
    time = 0
    # Time loop
    for it = 0:nt
        if (it == 11)
            t_tic = Base.time()
            niter = 0
        end
        dt = dx / (max(maximum(abs.(u)), maximum(abs.(v)), u_max)) / 4.1
        time = time + dt
        λx = dx / 2 / dt #x-direction signal speed for Lax-Friedrichs scheme
        λy = dy / 2 / dt #y-direction signal speed for Lax-Friedrichs scheme
        dt_dx, dt_dy = dt / dx, dt / dy
        λx_2, λy_2 = λx / 2, λy / 2

        #Lax-Friedrichs for Continuity
        @parallel compute_H!(H_new, H, u, v, dt_dx, dt_dy, size_H1_2, λx_2, λy_2)
        #Lax-Friedrichs for Momentum in x-direction
        @parallel compute_u!(u_new, u, H_new, H, dt_dx, dt_dy, size_H1_2, g)

        #Lax-Friedrichs for Momentum in y-direction
        @parallel compute_v!(v_new, v, H_new, H, u, dt_dx, dt_dy, size_H2_2, g)

        #No-flux boundary conditions
        @parallel 1:size(H, 2) bc_x!(H_new)
        u_new[1, :] .= 0
        u_new[end, :] .= 0
        v_new[:, 1] .= 0
        v_new[:, end] .= 0

        H .= H_new
        u .= u_new
        v .= v_new

        update_halo!(H)
        update_halo!(u)
        update_halo!(v)

        niter += 1
        # Visualize
        if do_visu && (it % nout == 0)
            H_inn .= H[2:end-1, 2:end-1]
            gather!(H_inn, H_v)
            z_max = maximum(H_v)
            z_min = minimum(H_v)
            if (me == 0)
                plot(Xi_g, Yi_g, Array(H_v)', st = :surface, xlims = (Xi_g[1], Xi_g[end]), ylims = (Yi_g[1], Yi_g[end]), zlims = (z_min, z_max), c = :blues, colorbar = false, xlabel = "Lx", ylabel = "Ly", zlabel = "water surface elevation (m)", title = "time = $(round(time, sigdigits=3)) s")
                frame(anim)
            end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc / niter                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                       # Effective memory throughput [GB/s]
    if (me == 0)
        @printf(
            "Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n",
            t_toc,
            round(T_eff, sigdigits = 3),
            niter
        )
    end
    if (do_visu && me == 0)
        if (dam1D_x && !dam2D)
            gif(anim, "plots/part-2/shallow_water_2D_xpu_1D-dam-x.gif", fps = 5)
        elseif (!dam1D_x && !dam2D)
            gif(anim, "plots/part-2/shallow_water_2D_xpu_1D-dam-y.gif", fps = 5)
        else 
            gif(anim, "plots/part-2/shallow_water_2D_xpu_2D-dam.gif", fps = 5) 
        end
    end
    # Retrieve the global domain
    finalize_global_grid()

    return Array(H), xc
end

# shallow_water_2D_xpu(; nx = 128, ny = 128, dam2D = true, do_visu = true)


