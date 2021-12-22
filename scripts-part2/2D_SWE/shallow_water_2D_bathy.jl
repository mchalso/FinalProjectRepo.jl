using Plots

"""
	shallow_water_2D_bathy(; nx, ny, dam_x=true, do_visu=true)

2D shallow water equations solver for an instantaneous dam break, incorporating 
bathymetry. Previous versions of this model could only utilize a constant bed elevation.
The calculations are now updated to include the effect of a non-constant bed elevation. 
In this test case for a dam break in the x-direction, the bed slopes from an elevation of
2.5 meters to -2.5 meters. The model domain has a length of 40 meters and width of 20 meters.  
Half of the domain has an initial water level of 5 meters, while the other half begins dry.
The effect of bathymetry is incorporated into the momentum calculations, 
as well as in the Lax-Friedrichs correction to the continuity equation.
Solution of momentum equations requires division by H. Therefore a minimum
H must be set throughout the domain, to avoid numerical instability.


# Arguments
    - `nx`: number of discretised cells for x dimension.
    - `ny`: number of discretised cells for y dimension.
    - `dam_x`: if true, 1D dam break in x-direction, else 1D dam break in y-direction. 
    - `do_visu`: if true, each physical time step will be ploted.
"""
@views function shallow_water_2D_bathy(;
    # Numerics
    nx = 512,
    ny = 256,
    # Modeling
    dam_x = true,
    # Visualisation
    do_visu = true)
    # Physics
    Lx, Ly = 40.0, 20.0
    g = 9.81
    u_max = 5 #from review of results with very small timesteps
    H_init = 5.0
    time = 0
    # Numerics
    nout = 100
    H_min = 1.0
    # Derived numerics
    dx, dy = Lx / nx, Ly / ny
    nt = 10000#cld(ttot, dt)
    xc, yc = LinRange(dx / 2, Lx - dx / 2, nx), LinRange(dy / 2, Ly - dy / 2, ny)
    # Initial Conditions
    η = zeros(Float64, nx, ny) .+ H_min
    if (dam_x)
        η[1:round(Int64(nx / 2)), :] = zeros(round(Int64(nx / 2)), ny) .+ H_init #1D dam break in x-direction
    else
        η[:, 1:round(Int64(ny / 2))] = zeros(Float64, nx, round(Int64(ny / 2))) .+ H_init #1D dam break in y direction
    end
    # Array initialisation
    #H = zeros(Float64, nx, ny) .+ H_init
    a = zeros(Float64, nx, ny)
    a[:,:]     .= a .+ H_init./2  .- xc .* H_init ./ Lx #bed elevation
    H = η .- a
    H[isless.(H,0.1)] .= 0.1
    H_new = copy(H)
    u = zeros(Float64, nx, ny)
    u_new = copy(u)
    v = zeros(Float64, nx, ny)
    v_new = copy(v)
    F = zeros(Float64, nx + 1, ny)
    G = zeros(Float64, nx, ny + 1)
    dHudt = zeros(Float64, nx-2, ny)
    gHdady = zeros(Float64, nx-2, ny)
    t_tic = 0.0
    niter = 0
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
        #Continuity
        F[2:end-1, :] .= 1 / 2 .* (H[1:end-1, :] .* u[1:end-1, :] .+ H[2:end, :] .* u[2:end, :]) .- λx / 2 .* (η[2:end, :] .- η[1:end-1, :])
        G[:, 2:end-1] .= 1 / 2 .* (H[:, 1:end-1] .* v[:, 1:end-1] .+ H[:, 2:end] .* v[:, 2:end]) .- λy / 2 .* (η[:, 2:end] .- η[:, 1:end-1])
        H_new[2:end-1, :] .= H[2:end-1, :] .- dt / dx .* (F[3:end-1, :] .- F[2:end-2, :]) .- dt / dy .* (G[2:end-1, 2:end] .- G[2:end-1, 1:end-1])
        H_new[isless.(H_new,0.1)] .= 0.1
        #Momentum in x-direction
        F[2:end-1, :] .= 1 / 2 .* ((H[1:end-1, :] .* ((u[1:end-1, :]) .^ 2) .+ 0.5 * g .* ((H[1:end-1, :]) .^ 2)) .+ (H[2:end, :] .* ((u[2:end, :]) .^ 2) .+ 0.5 * g .* ((H[2:end, :]) .^ 2)))
        G[:, 2:end-1] .= 1 / 2 .* ((H[:, 1:end-1] .* u[:, 1:end-1] .* v[:, 1:end-1]) .+ (H[:, 2:end] .* u[:, 2:end] .* v[:, 2:end]))
        gHdady .=.- g .* (0.5*(H[1:end-2,:].+H[3:end,:])) .* (1/2/dx.*(a[3:end,:].-a[1:end-2,:])) # term for effect of bathymetry
        dHudt .= .- 1 / dx .* (F[3:end-1, :] .- F[2:end-2, :]) .- 1 / dy .* (G[2:end-1, 2:end] .- G[2:end-1, 1:end-1]) .+ gHdady
        u_new[2:end-1, :] .= (1 / 2 .* (H[1:end-2, :] .* u[1:end-2, :] .+ H[3:end, :] .* u[3:end, :]) .+ dt .* dHudt) ./ H_new[2:end-1, :]
        #Momentum in y-direction
        F[2:end-1, :] .= 1 / 2 .* ((H[1:end-1, :] .* u[1:end-1, :] .* v[1:end-1, :]) .+ (H[2:end, :] .* u[2:end, :] .* v[2:end, :]))
        G[:, 2:end-1] .= 1 / 2 .* ((H[:, 1:end-1] .* ((v[:, 1:end-1]) .^ 2) .+ 0.5 * g .* ((H[:, 1:end-1]) .^ 2)) .+ (H[:, 2:end] .* ((v[:, 2:end]) .^ 2) .+ 0.5 * g .* ((H[:, 2:end]) .^ 2)))
        v_new[:, 2:end-1] .= (1 / 2 .* (H[:, 1:end-2] .* v[:, 1:end-2] .+ H[:, 3:end] .* v[:, 3:end]) .- dt / dx .* (F[2:end, 2:end-1] .- F[1:end-1, 2:end-1]) .- dt / dy .* (G[:, 3:end-1] .- G[:, 2:end-2])) ./ H_new[:, 2:end-1]
        #No-flux boundary conditions
        H_new[1, :] .= H_new[2, :]
        H_new[end, :] .= H_new[end-1, :]
        u_new[1, :] .= 0
        u_new[end, :] .= 0
        v_new[:, 1] .= 0
        v_new[:, end] .= 0

        H .= H_new
        u .= u_new
        v .= v_new
        η .= H .+ a

        niter += 1
        if do_visu && (it % nout == 0)
            #p1 = plot(xc, H[:, round(Int64(ny / ny))], xlims = (xc[1], xc[end]), ylims = (-3, 8),
            #    xlabel = "Lx at y=10 (m)", ylabel = "water surface elevation (m)", label = "h",
            #    title = "time = $(round(time, sigdigits=3)) s", linewidth = :1.0, framestyle = :box)
            #plot!(xc, u[:, round(Int64(ny / ny))], label = "u", linewidth = :1.0)
            #plot!(xc, a[:, round(Int64(ny / ny))], label = "bed", linewidth = :1.0)
            #plot!(xc, η[:, round(Int64(ny / ny))], label = "eta", linewidth = :1.0)
            #display(p1)
            optsη = (aspect_ratio=1.5, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), zlims=(-2.5, H_init),
                     clims=(0, H_init), c=:blues, xlabel="Lx", ylabel="Ly", zlabel="water surface",
                     title="Sloping bed, time = $(round(time, sigdigits=3))")
            optsa = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), zlims=(-2.5, H_init),
                     clims=(-5, H_init), c=:grays, xlabel="Lx", ylabel="Ly", zlabel="water surface",
                     title="time = $(round(time, sigdigits=3))")
            s1 = surface(xc, yc, a'; optsa...)
            surface!(xc, yc, η'; optsη...)
            display(s1)
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (3 * 2) / 1e9 * nx * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc / niter                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                       # Effective memory throughput [GB/s]
    printf("Time = %1.3f sec, T_eff = %1.4f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits = 3), niter)
    return H
end

#shallow_water_2D_bathy(; nx = 512, ny = 256, do_visu = true)
