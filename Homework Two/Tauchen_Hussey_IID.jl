#= ################################################################################################## 
    Tauchen-Hussey (1991) IID
=# ##################################################################################################

function tauchen_hussey_iid(N, σ, μ)
    """
    N: Number of discrete states
    σ: Standard deviation of the normal shocks
    μ: Mean of the normal shocks
    Returns: (nodes, weights) for the discretized IID shocks
    """
    nodes, weights = gausshermite(N)         # Gauss-Hermite nodes and weights
    nodes = nodes .* sqrt(2) .* σ .+ μ       # Apply scalar operations element-wise using dot syntax
    weights = weights ./ sqrt(π)             # Normalize weights
    return nodes, weights
end
