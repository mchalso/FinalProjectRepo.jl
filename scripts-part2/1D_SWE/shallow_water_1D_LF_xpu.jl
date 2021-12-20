const USE_xpu = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_xpu
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end
using Plots, Printf

macro Hx(ix) esc(:(H[$ix+2] - H[$ix])) end

"""
    compute_U!(u2, u, H, g, dt_dx, size_u1_2) 	

ParallelStencil kernel that computes the Lax-Friedrichs for the momentum in x-direction of the shallow water
equations.
"""
@parallel_indices (ix) function compute_U!(u2, u, H, g, dt_dx, size_u1_2)
	if (ix<=size_u1_2)
		u2[ix+1] = u[ix+1] + 1/2 * dt_dx * (-1/2 * (u[ix+2]*u[ix+2] - u[ix]*u[ix]) - g * @Hx(ix))
	end
    return
end

"""
	compute_H!(H2, H, dt_dx, u, size_H1_2)

ParallelStencil kernel that computes a time step of the shallow water 1D
process.
"""
@parallel_indices (ix) function compute_H!(H2, H, dt_dx, u, size_H1_2)
	if (ix<=size_H1_2)
		H2[ix+1] = 1/2 * (H[ix+2] + H[ix]) - 1/2 * dt_dx * ((H[ix+2] * u[ix+2]) - (H[ix] * u[ix])) #continuity with Lax-Friedrichs Method
	end
    return
end

"""
    shallow_water_1D_xpu(;do_visu=false)

1D shallow water equations solver for an instantaneous dam break.
The Lax-Friedrichs Method was applied to the continuity equation.
Geometry (length of 40 meters) and initial conditions (half of domain 
have initial water level of 5 meters, other half is dry) match 
BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel
As parameters, we can modify:
    - `do_visu`: if true, each physical time step will be ploted.
"""
@views function shallow_water_1D_xpu(;do_visu=false)
    # Physics
    Lx     = 40.0
    g      = 9.81
    u_max  = 6 #from review of results with very small timesteps
    ttot   = 20.0
    # Numerics
    nx     = 1024
    nout   = 100
    # Derived numerics
    dx     = Lx/nx
    dt     = dx/u_max/10 #stability condition of Lax-Friedrichs Method, further divided by 10 for stability of initially discontinuous wave front
    #println(dt)
    nt     = cld(ttot, dt)
    xc     = LinRange(dx/2, Lx-dx/2, nx)
    dt_dx  = dt / dx
    # Array initialisation
    H      = @zeros(nx)
    len = round(Int64(nx/2))
    H[1:len] = 5.0 .+ @zeros(len)
    H2     = copy(H)
    u      = @zeros(nx)
    u2     = copy(u)
    size_H1_2, size_u1_2 = size(H,1)-2, size(u,1)-2
    t_tic = 0.0; niter = 0
    # Time loop
    for it = 0:nt
        if (it==11) t_tic = Base.time(); niter = 0 end 
        @parallel compute_H!(H2, H, dt_dx, u, size_H1_2)
        @parallel compute_U!(u2, u, H, g, dt_dx, size_u1_2)
        H, H2 = H2, H # pointer swap
        u, u2 = u2, u # pointer swap

        # Boundary conditions
        H[1]        = H[2]
        H[end]      = H[end-1]
        u[1]        = -u[2]
        u[end]      = -u[end-1]
        niter += 1
        if do_visu && (it % nout == 0)
            p1=plot(xc, Array(H), xlims=(xc[1], xc[end]), ylims=(0, 10),
                xlabel="Lx (m)", ylabel="water surface elevation (m)", label="h",
                title="time = $(round(it*dt, sigdigits=3)) s, stability: $(round(maximum(abs.(u))*dt/dx,sigdigits=3)) /1", 
                linewidth=:1.0, framestyle=:box)
            #plot!(xc, u, label="u", linewidth=:1.0)
            display(p1)
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*2)/1e9*nx*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                      # Execution time per iteration [s]
    T_eff = A_eff/t_it                       # Effective memory throughput [GB/s]
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    return Array(H)
end

# shallow_water_1D_xpu(;do_visu=true)

# include("./shallow_water_1D_LF.jl")
# using Test
# @testset "Height H" begin
# 	H_xpu = shallow_water_1D_xpu()
# 	H_cpu = shallow_water_1D(;do_visu=false)
# 	@test H_xpu ≈ H_cpu 
# end;