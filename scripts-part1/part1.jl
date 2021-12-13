const USE_GPU = false
using LinearAlgebra
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using GLMakie, Makie
using Printf, ImplicitGlobalGrid
import MPI

GLMakie.activate!()

# macros to avoid array allocation
macro qx(ix,iy,iz) esc(:( -D_dx*(H[$ix+1,$iy+1,$iz+1] - H[$ix,$iy+1,$iz+1]) )) end
macro qy(ix,iy,iz) esc(:( -D_dy*(H[$ix+1,$iy+1,$iz+1] - H[$ix+1,$iy,$iz+1]) )) end
macro qz(ix,iy,iz) esc(:( -D_dz*(H[$ix+1,$iy+1,$iz+1] - H[$ix+1,$iy+1,$iz]) )) end

# Calculate the l2 norm of a matrix
norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))

@parallel_indices (ix,iy,iz) function compute_dual_time!(H, H2, Hold, dHdt, dHdt2, 
       _dt, dtau, _dx, _dy, _dz, D_dx, D_dy, D_dz, dmp, size_H1_2, size_H2_2, size_H3_2)

    if (ix<=size_H1_2 && iy<=size_H2_2 && iz<=size_H3_2)
        dHdt2[ix,iy,iz] = dmp * dHdt[ix,iy,iz] + 
            (H[ix+1,iy+1,iz+1] - Hold[ix+1,iy+1,iz+1]) * _dt - 
                                        ((@qx(ix+1,iy,iz) - @qx(ix,iy,iz))*_dx +
                                         (@qy(ix,iy+1,iz) - @qy(ix,iy,iz))*_dy +
                                         (@qz(ix,iy,iz+1) - @qz(ix,iy,iz))*_dz)

        H2[ix+1,iy+1,iz+1] = H[ix+1,iy+1,iz+1] + dtau * dHdt2[ix,iy,iz]
    end
    return
end

@views function diffusion_3D(; do_visu=false)
    # Physics
    Lx, Ly, Lz  = 10.0, 10.0, 10.0
    D       = 1.0
    ttot    = 1.0
    # Numerics
    nx, ny, nz  = 32, 32, 32 # number of grid points
    nout    = 10
    ϵ       = 1e-8
    maxIter = 1e5
    # Derived numerics
    me, dims = init_global_grid(nx, ny, nz)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = Lx/nx_g(), Ly/ny_g(), Lz/nz_g()
    dt      = 0.2
    _dt     = 1.0 / dt
    dtau    = (1.0/(max(dx, dy, dz)^2/D/8.1) + 1.0/dt)^-1
    nt      = cld(ttot, dt)
    xc, yc, zc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny), LinRange(dz/2, Lz-dz/2, nz)
    D_dx    = D/dx
    D_dy    = D/dy
    D_dz    = D/dz
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    # Array initialisation
    H0  = @zeros(nx,ny,nz)
    H0 .= 2.0*Data.Array([exp(-(x_g(ix,dx,H0)+dx/2 -Lx/2)^2 
                              -(y_g(iy,dy,H0)+dy/2 -Ly/2)^2 
                              -(z_g(iz,dz,H0)+dz/2 -Lz/2)^2) for ix=1:size(H0,1), 
                                                                 iy=1:size(H0,2),
                                                                 iz=1:size(H0,3)])
    H     = copy(H0)
    H2    = copy(H)
    Hold  = copy(H)
    Rh    = @zeros(nx-2,ny-2,nz-2)
    dHdt  = @zeros(nx-2,ny-2,nz-2)
    dHdt .= Data.Array([0.0 for ix=1:size(dHdt,1), iy=1:size(dHdt,2), iz=1:size(dHdt,3)])
    dHdt2 = copy(dHdt)
    size_H1_2, size_H2_2, size_H3_2 = size(H,1)-2, size(H,2)-2, size(H,3)-2
    # Preparation of visualisation
    if do_visu
        # Create directory if it does not exist
        if (~isdir("plots/part-1")) mkpath("plots/part-1") end

        if (me==0) ENV["GKSwstype"]="nul"; if isdir("viz2D_mxpu_out")==false mkdir("viz2D_mxpu_out") end; end

        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        H_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        H_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        Xi_g, Yi_g, Zi_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), 
                           LinRange(dy+dy/2, Ly-dy-dy/2, ny_v),
                           LinRange(dz+dz/2, Lz-dz-dz/2, nz_v)
    end

    t_tic = 0.0; niter = 0
    # Time loop
    t = 0.0; it = 0; ittot = 0;
    dmp = 1-29/nx
    while t<ttot
        if (it==1) t_tic = Base.time(); niter = 0 end
        iter = 0; err = 2*ϵ

        while err > ϵ && iter < maxIter
            # @hide_communication (8, 2) begin
                @parallel compute_dual_time!(H, H2, Hold, dHdt, dHdt2, _dt, dtau, 
                        _dx, _dy, _dz, D_dx, D_dy, D_dz, dmp, size_H1_2, size_H2_2, 
                        size_H3_2)
                H, H2 = H2, H
                dHdt, dHdt2 = dHdt2, dHdt
                update_halo!(H)
            # end
            iter += 1
            if (iter % nout == 0)
                err = norm_g(Rh)
            end
        end
        ittot += iter; it += 1; t += dt
        Hold .= H
        if isnan(err) error("Error is NaN!") end

        # Visualize
        if do_visu
            H_inn .= H[2:end-1,2:end-1,2:end-1]; gather!(H_inn, H_v)
            if (me==0)
                fig = Figure(resolution = (1200,1200))
                ax1 = Axis3(fig[1,1:2], perspectiveness = 0.5, azimuth = (it/nout)*0.2, 
                    elevation = 0.57, aspect = (1,1,1), title="Diffusion 3D domain (t=$(round(t*100)/100)s)")
                ax2 = Axis(fig[2,1], title="z=16 slice")
                ax3 = Axis(fig[2,2], title="x=16 slice")
                volume!(ax1, Xi_g, Yi_g, Zi_g, H_v; colormap=:viridis, transparency=true)
                heatmap!(fig[2,1], Xi_g, Yi_g, H_v[:,:,16]')
                heatmap!(fig[2,2], Xi_g, Yi_g, H_v[16,:,:]')
                # Add information
                save("plots/part-1/plot_$(it/nout).png", fig, px_per_unit = 2)
            end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = 2/1e9*nx_g()*ny_g()*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                          # Execution time per iteration [s]
    T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]
    if (me==0) @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter) end

    # Retrieve the global domain
    finalize_global_grid()
    return H, xc
end

#  diffusion_3D(; do_visu=false)
