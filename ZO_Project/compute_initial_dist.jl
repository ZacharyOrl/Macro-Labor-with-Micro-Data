############################################################
# Discretize a normal initial distribution onto a grid. 
############################################################
# Inputs: μ_0 - mean of the initial distribution, σ_0 - variance of the initial distribution, grid - a vector of grid points. 
# Outputs: dist - a vector of masses associated with each grid point. 
function compute_initial_dist(μ_0::Float64, σ_0::Float64, grid::Vector{Float64})
    n = length(grid) 

    dist =zeros(n)
    dist[1] = cdf(Normal(μ_0, σ_0), grid[1])

    for i = 2:n-1
        dist[i] = cdf(Normal(μ_0, σ_0), grid[i]) - cdf(Normal(μ_0, σ_0), grid[i - 1])
    end 

    dist[n] = 1 - cdf(Normal(μ_0, σ_0), grid[n - 1])

    return dist
end