# 1D shallow water equations solver for an instantaneous dam break.
# The Lax-Friedrichs Method was applied to the continuity equation.
# Geometry (length of 40 meters) and initial conditions (half of domain 
# have initial water level of 5 meters, other half is dry) match 
# BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel"

# juliap -O3 --check-bounds=no --math-mode=fast shallow_water_1D_LF_gpu
using Plots, Printf, CUDA

macro Hx(ix) esc(:(H[ix+2] - H[ix])) end

function compute_U!(u2, u, H, g, dt_dx, size_u1_2)
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	if (ix<=size_u1_2)
		u2[ix+1] = u[ix+1] + 1/2 * dt_dx * (-1/2 * (u[ix+2]^2 - u[ix]^2) - g*@Hx(ix))
	end
    return
end

function compute_H!(H2, H, dt_dx, u, size_H1_2)
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	if (ix<=size_H1_2)
		H2[ix+1] = 1/2 * (H[ix+2] + H[ix]) - 1/2 * dt_dx * ((H[ix+2] * u[ix+2]) - (H[ix] * u[ix])) #continuity with Lax-Friedrichs Method
	end
    return
end

@views function shallow_water_1D()
    # Physics
    Lx     = 40.0
    g      = 9.81
    u_max  = 6 #from review of results with very small timesteps
    ttot   = 20.0
    # Numerics
    BLOCKX  = 32
    GRIDX   = 32
    nx     = BLOCKX*GRIDX
    nout   = 100
    # Derived numerics
    dx     = Lx/nx
    dt     = dx/u_max/10 #stability condition of Lax-Friedrichs Method, further divided by 10 for stability of initially discontinuous wave front
    #println(dt)
    nt     = cld(ttot, dt)
    xc     = LinRange(dx/2, Lx-dx/2, nx)
    dt_dx  = dt / dx
    # Array initialisation
    H      = CUDA.zeros(Float64, nx  )
    H[1:round(Int64(nx/2))] = 5.0 .+ CUDA.zeros(Float64, round(Int64(nx/2)))
    H2     = copy(H)
    u      = CUDA.zeros(Float64, nx  )
    u2     = copy(u)
    cuthreads = (BLOCKX, 1)
    cublocks  = (GRIDX,  1)
    size_H1_2, size_u1_2 = size(H,1)-2, size(u,1)-2
    t_tic = 0.0; niter = 0
    # Time loop
    for it = 0:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        @cuda blocks=cublocks threads=cuthreads compute_H!(H2, H, dt_dx, u, size_H1_2)
        synchronize()
        H, H2 = H2, H # pointer swap
        H[1]        = H[2]
        H[end]      = H[end-1]
        @cuda blocks=cublocks threads=cuthreads compute_U!(u2, u, H, g, dt_dx, size_u1_2)
        synchronize()
        u, u2 = u2, u # pointer swap
        u[1]        = -u[2]
        u[end]      = -u[end-1]
        niter += 1
        if it % nout == 0
            p1=plot(xc, Array(H), xlims=(xc[1], xc[end]), ylims=(0, 10),
                xlabel="Lx (m)", ylabel="water surface elevation (m)", label="h",
                title="time = $(round(it*dt, sigdigits=3)) s, stability: $(round(maximum(abs.(u))*dt/dx,sigdigits=3)) /1", 
                linewidth=:1.0, framestyle=:box)
            #plot!(xc, u, label="u", linewidth=:1.0)
            display(p1)
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*2)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                      # Execution time per iteration [s]
    T_eff = A_eff/t_it                       # Effective memory throughput [GB/s]
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    return
end

shallow_water_1D()