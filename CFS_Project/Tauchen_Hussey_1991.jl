#= ##################################################################################################
    Tauchen-Hussey (1991) AR(1) Process (Non-IID)
    Approximates z_{t+1} = ρ z_t + ε_{t+1}, with ε ~ N(0, σ²)
=# ##################################################################################################

using FastGaussQuadrature
using LinearAlgebra

function tauchen_hussey(N, ρ, σ, μ=0.0)
    """
    N   : Number of grid points
    ρ   : Persistence of the AR(1) process
    σ   : Std. dev. of the i.i.d. shocks (ε)
    μ   : Unconditional mean (default = 0.0)
    
    Returns:
        z_grid   : Vector of discretized states
        P        : Transition matrix of size N×N
    """
    
    # Step 1: Get Gauss-Hermite nodes and weights for standard normal
    nodes, weights = gausshermite(N)
    weights        ./= sqrt(π)                # Normalize weights
    nodes          .*= sqrt(2)                # Rescale for standard normal

    # Step 2: Adjust nodes for unconditional distribution of z_t
    σ_z             = σ / sqrt(1 - ρ^2)       # Std. dev. of stationary z
    z_grid          = μ .+ σ_z .* nodes       # Grid of z_t values

    # Step 3: Compute transition matrix P
    P = zeros(N, N)
    for i in 1:N
        # Conditional mean of z_{t+1} given z_t = z_grid[i]
        cond_mean = μ + ρ * (z_grid[i] - μ)

        # Approximate conditional distribution using GH quadrature
        for j in 1:N
            f = exp(-(z_grid[j] - cond_mean)^2 / (2 * σ^2))
            P[i, j] = weights[j] * f
        end

        # Normalize row to sum to 1
        P[i, :] ./= sum(P[i, :])
    end

    return z_grid, P
end
