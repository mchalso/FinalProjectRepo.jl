# 2D shallow water equations solver for an instantaneous dam break.
# The Lax-Friedrichs Method was used to stabilize the numerical scheme.
# Geometry (length of 40 meters, width of 20 meters) and initial conditions 
# (half of domain has initial water level of 5 meters, other half is dry) 
# matches BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel."
# Solution of momentum equations requires division by H. Therefore a minimum
# H must be set throughout the domain, to avoid numerical instability.

using Plots

@views function shallow_water_2D(;
    # Numerics
    nx = 512,
    ny = 256,
    # Visualisation
    do_viz=true)
    # Physics
    Lx, Ly = 40.0, 20.0
    g      = 9.81
    u_max  = 5 #from review of results with very small timesteps
    H_init = 5.0
    time   = 0
    # Numerics
    nout   = 100
    H_min  = 1.0
    # Derived numerics
    dx, dy = Lx/nx, Ly/ny
    nt     = 10000#cld(ttot, dt)
    xc, yc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    # Initial Conditions
    H      = zeros(Float64, nx  , ny  ) .+ H_min
    H[1:round(Int64(nx/2)), :] = zeros(Float64, round(Int64(nx/2)), ny) .+ H_init #1D dam break in x-direction
    # H[:,1:round(Int64(ny/2))] = zeros(Float64, nx, round(Int64(ny/2))) .+ H_init #1D dam break in y direction
    #H[1:round(Int64(nx/2)), 1:round(Int64(ny/2))] = zeros(Float64, round(Int64(nx/2)), round(Int64(ny/2))) .+ H_init #2D dam break
    #H[round(Int64(3*nx/8)):round(Int64(5*nx/8)), round(Int64(3*ny/8)):round(Int64(5*ny/8))] = zeros(Float64, round(Int64(nx/4))+1, round(Int64(ny/4))+1) .+ H_init #2D water column
    # Array initialisation
    H_new  = copy(H)
    u      = zeros(Float64, nx  , ny  )
    u_new  = copy(u)
    v      = zeros(Float64, nx  , ny  )
    v_new  = copy(v)
    F      = zeros(Float64, nx+1, ny  )
    G      = zeros(Float64, nx  , ny+1)
    # Time loop
    for it = 0:nt
        dt = dx/(max(maximum(abs.(u)),maximum(abs.(v)),u_max))/4.1;
        time = time + dt
        λx     = dx/2/dt #x-direction signal speed for Lax-Friedrichs scheme
        λy     = dy/2/dt #y-direction signal speed for Lax-Friedrichs scheme
        #Continuity
        F[2:end-1,:]     .= 1/2 .*(H[1:end-1,:].*u[1:end-1,:] .+ H[2:end  ,:].*u[2:end  ,:]) .- λx/2 .*(H[2:end  ,:] .- H[1:end-1,:])
        G[:,2:end-1]     .= 1/2 .*(H[:,1:end-1].*v[:,1:end-1] .+ H[:,2:end  ].*v[:,2:end  ]) .- λy/2 .*(H[:,2:end  ] .- H[:,1:end-1])
        H_new[2:end-1,:] .= H[2:end-1,:] .- dt/dx .*(F[3:end-1,:] .- F[2:end-2,:]) .- dt/dy .*(G[2:end-1,2:end] .- G[2:end-1,1:end-1])
        #Momentum in x-direction
        F[2:end-1,:]     .= 1/2 .* ( (H[1:end-1,:].*((u[1:end-1,:]).^2) .+ 0.5*g.*((H[1:end-1,:]).^2)) .+ (H[2:end  ,:].*((u[2:end  ,:]).^2) .+ 0.5*g.*((H[2:end  ,:]).^2)) )
        G[:,2:end-1]     .= 1/2 .* ( (H[:,1:end-1].*u[:,1:end-1].*v[:,1:end-1]) .+ (H[:,2:end  ].*u[:,2:end  ].*v[:,2:end  ]) )
        u_new[2:end-1,:] .= (1/2 .*(H[1:end-2,:].*u[1:end-2,:] .+ H[3:end  ,:].*u[3:end  ,:]) .- dt/dx .*(F[3:end-1,:] .- F[2:end-2,:]) .- dt/dy .*(G[2:end-1,2:end  ] .- G[2:end-1,1:end-1]) ) ./ H_new[2:end-1,:]
        #Momentum in y-direction
        F[2:end-1,:]     .= 1/2 .* ( (H[1:end-1,:].*u[1:end-1,:].*v[1:end-1,:]) .+ (H[2:end  ,:].*u[2:end  ,:].*v[2:end  ,:]) )
        G[:,2:end-1]     .= 1/2 .* ( (H[:,1:end-1].*((v[:,1:end-1]).^2) .+ 0.5*g.*((H[:,1:end-1]).^2)) .+ (H[:,2:end  ].*((v[:,2:end  ]).^2) .+ 0.5*g.*((H[:,2:end  ]).^2)) )
        v_new[:,2:end-1] .= (1/2 .*(H[:,1:end-2].*v[:,1:end-2] .+ H[:,3:end  ].*v[:,3:end  ]) .- dt/dx .*(F[2:end  ,2:end-1] .- F[1:end-1,2:end-1]) .- dt/dy .*(G[:,3:end-1] .- G[:,2:end-2]) ) ./ H_new[:,2:end-1]
        #No-flux boundary conditions
        H_new[1,:]   .= H_new[2,:]
        H_new[end,:] .= H_new[end-1,:]
        u_new[1,:]   .= 0
        u_new[end,:] .= 0
        v_new[:,1]   .= 0
        v_new[:,end] .= 0

        H .= H_new
        u .= u_new
        v .= v_new

        if do_viz && (it % nout == 0)
            p1=plot(xc, H[:,round(Int64(ny/ny))], xlims=(xc[1], xc[end]), ylims=(-3, 8),
               xlabel="Lx at y=10 (m)", ylabel="water surface elevation (m)", label="h",
                title="time = $(round(time, sigdigits=3)) s", linewidth=:1.0, framestyle=:box)
            plot!(xc, u[:,round(Int64(ny/ny))], label="u", linewidth=:1.0)
            plot!(xc, v[:,round(Int64(ny/ny))], label="v", linewidth=:1.0)
            display(p1)
            #opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), zlims=(H_min, H_init),
            #        clims=(H_min, H_init), c=:blues, xlabel="Lx", ylabel="Ly", zlabel="water surface",
            #        title="time = $(round(time, sigdigits=3))")
            #display(surface(xc, yc, H'; opts...))
        end
    end
    return H
end

# shallow_water_2D()