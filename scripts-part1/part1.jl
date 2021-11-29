# Part 1 of final project: Diffusion equation

@views function init()
    nx_g = 30             # number of global grid points
    Xc_g = zeros(nx_g, 1) # global coord vector
    H_g  = zeros(nx_g, 1) # global solution as obtained by implicitGlobalGrid's `gather!()`
    inds = Int.(ceil.(LinRange(1, length(Xc_g), 12)))
    Xc_g[inds] .= [0.46875, 1.40625, 2.34375, 2.96875, 3.90625, 4.84375, 5.46875, 6.40625, 7.34375, 7.96875, 8.90625, 9.53125]
    H_g[inds]  .= [1.2981288953788742e-6, 6.258528479372478e-6, 1.4557797727681202e-5, 2.2141755782992116e-5, 3.3990701348812344e-5, 4.0487959896720236e-5, 3.931867494216368e-5, 3.028472793911268e-5, 1.819088664977141e-5, 1.1341281028367691e-5, 4.326911919257118e-6, 1.2981288953788742e-6]
    return Xc_g, H_g
end

Xc_g, H_g = init();
