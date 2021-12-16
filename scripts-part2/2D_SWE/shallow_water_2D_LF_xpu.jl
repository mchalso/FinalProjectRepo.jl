# 2D shallow water equations solver for an instantaneous dam break.
# The Lax-Friedrichs Method was applied to the continuity equation.
# Geometry (length of 40 meters, width of 20 meters) and initial conditions 
# (half of domain has initial water level of 5 meters, other half is dry) 
# matches BASEMENT version 2.8 Test Case H_1 "Dam break in a closed channel."
# Solution of momentum equations requires division by H. Therefore a minimum
# H must be set throughout the domain, to avoid numerical instability.

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf

@parallel_indices (ix,iy) function compute_FG1!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1)
	if (ix<=size_H1_1 && iy<=size(H,2))
		F[ix+1,iy] = 1/2 * (H[ix,iy] * u[ix,iy] + H[ix+1,iy]*u[ix+1,iy]) - λx_2 * (H[ix+1,iy] - H[ix,iy]) 
	end
	if (ix<=size(H,1) && iy<=size_H2_1) 
		G[ix,iy+1] = 1/2 * (H[ix,iy] * v[ix,iy] + H[ix,iy+1]*v[ix,iy+1]) - λy_2 * (H[ix,iy+1] - H[ix,iy]) 
	end
	return
end

@parallel_indices (ix,iy) function compute_H!(H_new, H, F, G, dt_dx, dt_dy, size_H1_2) 
	if (ix<=size_H1_2 && iy<=size(H,2))
		H_new[ix+1,iy] = H[ix+1,iy] - dt_dx * (F[ix+2,iy]-F[ix+1,iy]) - dt_dy * (G[ix+1,iy+1] - G[ix+1,iy]) 
	end
	return
end

@parallel_indices (ix,iy) function compute_FG2!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1, g)
	if (ix<=size_H1_1 && iy<=size(H,2))
		F[ix+1,iy] = 1/2 * ((H[ix,iy] * u[ix,iy]^2) + 0.5*g*H[ix,iy]^2 + H[ix+1,iy]*u[ix+1,iy]^2 + 0.5*g*H[ix+1,iy]^2)
					 - λx_2 * (H[ix+1,iy]*u[ix+1,iy] - H[ix,iy]*u[ix,iy])
	end
	if (ix<=size(H,1) && iy<=size_H2_1) 
		G[ix,iy+1] = 1/2 * (H[ix,iy] * u[ix,iy]*v[ix,iy] + H[ix,iy+1]*u[ix,iy+1]*v[ix,iy+1])
					 - λy_2 * (H[ix,iy+1]*u[ix,iy+1] - H[ix,iy]*u[ix,iy]) 
	end
	return
end

@parallel_indices (ix,iy) function compute_u!(u_new, u, H_new, H, F, G, dt_dx, dt_dy, size_H1_2) 
	if (ix<=size_H1_2 && iy<=size(H,2))
		u_new[ix+1,iy] = (1/2 * (H[ix,iy] * u[ix,iy] + H[ix+2,iy]*u[ix+2,iy])
					 - dt_dx * (F[ix+2,iy]-F[ix+1,iy]) - dt_dy * (G[ix+1,iy+1] - G[ix+1,iy+1])
					) / H_new[ix+1,iy] 
	end
	return
end

@parallel_indices (ix,iy) function compute_FG3!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1, g)
	if (ix<=size_H1_1 && iy<=size(H,2))
		F[ix+1,iy] = 1/2 * (H[ix,iy] * u[ix,iy]*v[ix,iy] + H[ix+1,iy]*u[ix+1,iy]*v[ix+1,iy])
						- λx_2 * (H[ix+1,iy]*v[ix+1,iy] - H[ix,iy]*v[ix,iy])
	end
	if (ix<=size(H,1) && iy<=size_H2_1) 
		G[ix,iy+1] = 1/2 * ((H[ix,iy] * v[ix,iy]^2) + 0.5*g*H[ix,iy]^2 + H[ix,iy+1]*v[ix,iy+1]^2 + 0.5*g*H[ix,iy+1]^2)
					 - λy_2 * (H[ix,iy+1]*v[ix,iy+1] - H[ix,iy]*v[ix,iy]) 
	end
	return
end

@parallel_indices (ix,iy) function compute_v!(v_new, v, H_new, H, F, G, dt_dx, dt_dy, size_H2_2) 
	if (ix<=size(H,1) && iy<=size_H2_2)
		v_new[ix,iy+1] = (1/2 * (H[ix,iy] * v[ix,iy] + H[ix,iy+2]*v[ix,iy+2])
						- dt_dx * (F[ix+1,iy+1]-F[ix,iy+1]) - dt_dy * (G[ix,iy+2] - G[ix,iy+1])
					) / H_new[ix,iy+1]
	end
	return
end

@views function shallow_water_2D_gpu(;
	# Numerics
	nx = 512,
	ny = 256,
	# Visualisation
	do_viz=false)
	# Physics
	Lx, Ly = 40.0, 20.0
	g      = 9.81
	u_max  = 4 #from review of results with very small timesteps
	v_max  = 0 #dam break occurs only in x-direction
	ttot   = 20
	H_init = 5.0
	# Numerics
	nout   = 100
	H_min  = 1.0
	# Derived numerics
	dx, dy = Lx/nx, Ly/ny
	dt     = min(dx/2/u_max, dy/2/v_max)/4 #stability condition of Lax-Friedrichs Method, further divided for stability of initially discontinuous wave front
	λx     = dx/2/dt
	λy     = dy/2/dt
	nt     = cld(ttot, dt)
	xc, yc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
	λx_2, λy_2 = λx/2, λy/2
	dt_dx, dt_dy = dt/dx, dt/dy
	# Array initialisation
	H      = @zeros(nx  , ny  ) .+ H_min
	H[1:round(Int64(nx/2)), :] = @zeros(round(Int64(nx/2)), ny) .+ H_init #1D dam break
	#H[1:round(Int64(nx/2)), 1:round(Int64(ny/2))] = @zeros(round(Int64(nx/2)), round(Int64(ny/2))) .+ H_init #2D dam break
	H_new  = copy(H)
	u      = @zeros(nx  , ny  )
	u_new  = copy(u)
	v      = @zeros(nx  , ny  )
	v_new  = copy(v)
	F      = @zeros(nx+1, ny  )
	G      = @zeros(nx  , ny+1)
	size_H1_2, size_H2_2 = size(H,1)-2, size(H,2)-2
	size_H1_1, size_H2_1 = size(H,1)-1, size(H,2)-1
	# Time loop
	for it = 0:nt
		#Lax-Friedrichs for Continuity
		# F[2:end-1,:]     .= 1/2 .*(H[1:end-1,:].*u[1:end-1,:] .+ H[2:end  ,:].*u[2:end  ,:]) .- λx_2 .*(H[2:end  ,:] .- H[1:end-1,:])
		@parallel compute_FG1!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1)
		#   H_new[2:end-1,:] .= H[2:end-1,:] .- dt/dx .*(F[3:end-1,:] .- F[2:end-2,:]) .- dt/dy .*(G[2:end-1,2:end] .- G[2:end-1,1:end-1])
		@parallel compute_H!(H_new, H, F, G, dt_dx, dt_dy, size_H1_2) 
		H_new[1,:]   .= H_new[2,:]
		H_new[end,:] .= H_new[end-1,:]

		#Lax-Friedrichs for Momentum in x-direction
		# F[2:end-1,:]     .= 1/2 .* ( (H[1:end-1,:].*((u[1:end-1,:]).^2) .+ 0.5*g.*((H[1:end-1,:]).^2))
		# 				.+ (H[2:end  ,:].*((u[2:end  ,:]).^2) .+ 0.5*g.*((H[2:end  ,:]).^2)) )
		# 			.- λx_2 .* ((H[2:end,:].*u[2:end,:]) .- (H[1:end-1,:].*u[1:end-1,:]))
		# G[:,2:end-1]     .= 1/2 .* ( (H[:,1:end-1].*u[:,1:end-1].*v[:,1:end-1])
		# 				.+ (H[:,2:end  ].*u[:,2:end  ].*v[:,2:end  ]) )
		# 			.- λy_2 .*  (H[:,2:end  ].*u[:,2:end  ] .- H[:,1:end-1].*u[:,1:end-1])
		@parallel compute_FG2!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1, g)
		for iy=1:size(H,2)
			for ix=1:size_H1_2
				u_new[ix+1,iy] = (1/2 * (H[ix,iy] * u[ix,iy] + H[ix+2,iy]*u[ix+2,iy])
						 	- dt_dx * (F[ix+2,iy]-F[ix+1,iy]) - dt_dy * (G[ix+1,iy+1] - G[ix+1,iy+1])
							) / H_new[ix+1,iy]

			end
		end
		# u_new[2:end-1,:] .= (1/2 .*(H[1:end-2,:].*u[1:end-2,:] .+ H[3:end  ,:].*u[3:end  ,:])
		# 				.- dt_dx .*(F[3:end-1,:] .- F[2:end-2,:]) .- dt_dy .*(G[2:end-1,2:end] .- G[2:end-1,1:end-1]) 
		# 			) ./ H_new[2:end-1,:]
		@parallel compute_u!(u_new, u, H_new, H, F, G, dt_dx, dt_dy, size_H1_2) 
		u_new[1,:]   .= 0
		u_new[end,:] .= 0

		#Lax-Friedrichs for Momentum in y-direction
		# F[2:end-1,:] .= 1/2 .* ( (H[1:end-1,:].*u[1:end-1,:].*v[1:end-1,:])
		# 				.+ (H[2:end  ,:].*u[2:end  ,:].*v[2:end  ,:]) )
		# 		.- λx_2 .*  (H[2:end  ,:].*v[2:end  ,:] .- H[1:end-1,:].*v[1:end-1,:])
		# G[:,2:end-1] .= 1/2 .* ( (H[:,1:end-1].*((v[:,1:end-1]).^2) .+ 0.5*g.*((H[:,1:end-1]).^2))
		# 				.+ (H[:,2:end  ].*((v[:,2:end  ]).^2) .+ 0.5*g.*((H[:,2:end  ]).^2)) )
		# 		.- λy_2 .* ((H[:,2:end].*v[:,2:end]) .- (H[:,1:end-1].*v[:,1:end-1]))
		@parallel compute_FG3!(F,G,H,u,v,λx_2, λy_2,size_H1_1, size_H2_1, g)
		# v_new[:,2:end-1] .= (1/2 .*(H[:,1:end-2] .* v[:,1:end-2] .+ H[:,3:end  ] .* v[:,3:end  ]) 
		# 				.- dt_dx.*(F[2:end  ,2:end-1] .- F[1:end-1,2:end-1]) .- dt_dy .*(G[:,3:end-1] .- G[:,2:end-2])
		# 			) ./ H_new[:,2:end-1]
		@parallel compute_v!(v_new, v, H_new, H, F, G, dt_dx, dt_dy, size_H2_2) 
		v_new[:,1]   .= 0
		v_new[:,end] .= 0
		
		H .= H_new
		u .= u_new
		v .= v_new

		if do_viz && (it % nout == 0)
			p1=plot(xc, H[:,round(Int64(ny/2))], xlims=(xc[1], xc[end]), ylims=(0, 10),
			xlabel="Lx at y=10 (m)", ylabel="water surface elevation (m)", label="h",
			title="time = $(round(it*dt, sigdigits=3)) s", linewidth=:1.0, framestyle=:box)
			plot!(xc, u[:,round(Int64(ny/2))], label="u", linewidth=:1.0)
			plot!(xc, v[:,round(Int64(ny/2))], label="v", linewidth=:1.0)
			display(p1)
			opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), zlims=(H_min, H_init),
				clims=(H_min, H_init), c=:blues, xlabel="Lx", ylabel="Ly", zlabel="water surface",
				title="time = $(round(it*dt, sigdigits=3))")
			display(surface(xc, yc, H'; opts...))
		end
	end
	return H
end

shallow_water_2D_gpu(;nx=128, ny=64, do_viz=false)

include("./shallow_water_2D_LF.jl")
using Test
@testset "Height H" begin
	H_gpu = shallow_water_2D_gpu(;nx=128, ny=64, do_viz=false)
	H_cpu = shallow_water_2D(;nx=128, ny=64, do_viz=false)
	@test H_gpu ≈ H_cpu 
end;