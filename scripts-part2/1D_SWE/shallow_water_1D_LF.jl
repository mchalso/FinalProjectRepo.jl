# 1D shallow water equations solver for an instantaneous dam break.
# The Lax-Friedrichs Method was applied to the continuity equation.
# Geometry (length of 40 meters) and initial conditions (half of domain 
# have initial water level of 5 meters, other half is dry) match 
# BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel"

using Plots

@views function shallow_water_1D(;do_visu=false)
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
    # Array initialisation
    H      = zeros(Float64, nx  )
    H[1:round(Int64(nx/2))] = 5.0 .+ zeros(Float64, round(Int64(nx/2)))
    u      = zeros(Float64, nx  )
    dHdx   = zeros(Float64, nx-1)
    dudx   = zeros(Float64, nx-1)
    dudt   = zeros(Float64, nx-1)
    # Time loop
    for it = 0:nt
        dHdx       .= diff(H) ./ dx
        dudx       .= diff(u) ./ dx
        dudt       .= -1/2 .* (u[1:end-1].+u[2:end]) .* dudx .- (g .* dHdx) #momentum
        H[2:end-1] .= 1/2 .* (H[3:end].+H[1:end-2]) .- dt / 2 / dx .* ((H[3:end] .* u[3:end]) .- (H[1:end-2] .* u[1:end-2])) #continuity with Lax-Friedrichs Method
        H[1]        = H[2]
        H[end]      = H[end-1]
        u[2:end-1] .= u[2:end-1] .+ 1/2 .* (dudt[1:end-1] .+ dudt[2:end]) .* dt
        u[1]        = -u[2]
        u[end]      = -u[end-1]
        if do_visu && (it % nout == 0)
            p1=plot(xc, H, xlims=(xc[1], xc[end]), ylims=(0, 10),
                xlabel="Lx (m)", ylabel="water surface elevation (m)", label="h",
                title="time = $(round(it*dt, sigdigits=3)) s, stability: $(round(maximum(abs.(u))*dt/dx,sigdigits=3)) /1", 
                linewidth=:1.0, framestyle=:box)
            #plot!(xc, u, label="u", linewidth=:1.0)
            display(p1)
        end
    end
    return H
end

# shallow_water_1D(do_visu=true)