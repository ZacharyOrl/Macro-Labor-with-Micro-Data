#= ################################################################################################## 
    Tauchen (1986)
=# ##################################################################################################

using Distributions

function tauchen(N::Int, ρ::Float64, σ::Float64, μ::Float64=0.0, m::Float64=3.0)
    """
    Discretizes an AR(1) process: y' = μ + ρ * y + ε, ε ~ N(0, σ^2)

    Arguments:
        N  : Number of grid points
        ρ  : AR(1) persistence parameter
        σ  : Standard deviation of ε
        μ  : Mean of the process (default = 0)
        m  : Width parameter for grid range (default = 3 standard deviations)

    Returns:
        y_grid   : Vector of grid points
        P        : Transition matrix (N x N)
    """

    # Standard deviation of stationary distribution
    σ_y = σ / sqrt(1 - ρ^2)

    # Grid for y: centered at μ, spanning ±m*σ_y
    y_min = μ - m * σ_y
    y_max = μ + m * σ_y
    y_grid = range(y_min, y_max, length=N) |> collect
    Δ = y_grid[2] - y_grid[1]  # Step size

    # Transition probability matrix
    P = zeros(N, N)

    for j in 1:N
        for k in 1:N
            # Mean of next-period shock conditional on y_j
            μ_cond = μ + ρ * (y_grid[j] - μ)

            if k == 1
                P[j, k] = cdf(Normal(μ_cond, σ), y_grid[k] + Δ/2)
            elseif k == N
                P[j, k] = 1 - cdf(Normal(μ_cond, σ), y_grid[k] - Δ/2)
            else
                P[j, k] = cdf(Normal(μ_cond, σ), y_grid[k] + Δ/2) -
                          cdf(Normal(μ_cond, σ), y_grid[k] - Δ/2)
            end
        end
    end

    return y_grid, P
end
