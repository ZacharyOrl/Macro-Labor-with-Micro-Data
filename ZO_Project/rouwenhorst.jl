#  Matches three moments: the expectation - autocorrelation and variance 
#  you want to match and three parameters - N, σ and ρ. 
#  this means the optimal GMM weighting matrix is identity 

# Inputs[variance of process you want to discretize (not the variance of the shock), autocorrelation of that process, N = number of gridpoints; Weighting Matrix]
# Outputs[gridpoint vector, transition probability matrix]
function rouwenhorst(σ::Float64, ρ::Float64, N::Int64; W::Matrix{Float64} = Matrix{Float64}(I, 3, 3))

    # --- Step 1: Compute theoretical moments from candidate parameters
    function compute_moments(params::Vector{Float64}, N::Int64)
        p, q, ψ = params
        s = (1 - p) / (2 - (p + q))
        expec = (q - p) * ψ / (2 - (p + q))
        corr = p + q - 1
        var = ψ^2 * (1 - 4s*(1 - s) + (4s*(1 - s))/(N - 1))
        return [expec, corr, var]
    end

    # --- Step 2: GMM objective function
    function objective(params::Vector{Float64}, W::Matrix{Float64}, sample_moments::Vector{Float64}, N::Int64)
        model_moments = compute_moments(params, N)
        g_hat = model_moments .- sample_moments
        return g_hat' * W * g_hat
    end

    # --- Step 3: Estimate optimal (p, q, ψ) using GMM
    function GMM(σ::Float64, ρ::Float64, N::Int64, W::Matrix{Float64})
        sample_moments = [0.0, ρ, σ]
        params_init = 0.5 * ones(3)
        result = optimize(p -> objective(p, W, sample_moments, N), params_init, BFGS())
        return Optim.minimizer(result)
    end

    # --- Step 4: Construct transition matrix
    function compute_transition_matrix(p_hat::Vector{Float64}, N::Int64)
        p, q = p_hat[1], p_hat[2]
        T = [p 1 - p; 1 - q q]

        for i in 2:N-1
            zero_vec = zeros(i, 1)
            T = p   * [T zero_vec; zero_vec' 0] +
                (1 - p) * [zero_vec T; 0 zero_vec'] +
                (1 - q) * [zero_vec' 0; T zero_vec] +
                q   * [0 zero_vec'; zero_vec T]

            for n = 2:size(T, 2) - 1
                T[n, :] *= 0.5
            end
        end
        return T
    end

    # --- Step 5: Generate evenly spaced grid
    function generate_grid(p_hat::Vector{Float64}, N::Int64)
        ψ = p_hat[3]
        return collect(range(start = -ψ, stop = ψ, length = N))
    end

    # === Call steps ===
    p_hat = GMM(σ, ρ, N, W)
    P = compute_transition_matrix(p_hat, N)
    grid = generate_grid(p_hat, N)

    return grid, P
end
