################################################################
# Econ 810: Spring 2025 Advanced Macroeconomics 
# Estimate variance of persistent and transitory shocks
################################################################
# Add Packages 
using Statistics, Optim, LinearAlgebra, DataFrames, CSV, ForwardDiff
################################################################
# Put your path to the dataset "psid_cleaned.csv" below 
indir = "$outdir_parameters"

# Put the outdirectory below if desired 
outdir = indir
cd(indir)
################################################################
# Parameters 
ρ = 0.97
################################################################
# Load data 
data = CSV.File("psid_cleaned.csv")
y = Array(data.residual_component)
x = Array(data.person_id)
df = DataFrame(residual_income = y, person_id = x)

################################################################
# GMM Functions
################################################################
# This function estimates the two BPP moments: 
# Cov(y_tilde_t,y_tilde_{t+1})
# Cov(y_tilde_t,ρ^2 * y_tilde_{t-1} + ρ * y_tilde_{t} + y_tilde_{t+1})
# for each individual in the panel. 
function est_moments(df::DataFrame,ρ::Float64)
    # Compute moments (variance and covariance) for each individual.
    id_list = unique(df.person_id)  # Groups to compute covariance within
    N = length(id_list)
    out = zeros(N, 2)

    for i in 1:N
        id_i = id_list[i]  # Identifier for the i-th person

        # Extract the data specific to that person. 
        sub_data = filter(row -> row.person_id == id_i, df).residual_income
        n = size(sub_data, 1)

        # Compute the specific type of difference suggested by the slides: 
        y_tilde = sub_data[2:n] .- ρ .* sub_data[1:n-1] # y_tilde will be a n-1 vector.

        # Compute the other quantity suggested by the slides / BPP: 
        sum_y_tilde = ρ^2 .* y_tilde[1:n-3] .+ ρ .* y_tilde[2:n-2] .+ y_tilde[3:n-1] # sum_y_tilde will be a n-3 vector
        m = mean(y_tilde) 
        m_sum = mean(sum_y_tilde)

        # Use biased estimator
        cov = (1 / (n-1)) * ((y_tilde[2] -  m) * (y_tilde[1] -  m))
        cov_sum = 0.0 

        for j = 3:n-1
            cov += (1 / (n-1)) * (y_tilde[j] - m) * (y_tilde[j-1] - m)
            cov_sum += (1 / (n-3)) * (y_tilde[j-1] - m) * (sum_y_tilde[j-2] - m_sum)
        end
        out[i, 1] = cov
        out[i, 2] = cov_sum
    end

    # Return a matrix where each row is [covariance,covariance with the sum] for an individual
    cov_hat = out[:, 1]
    cov_sum_hat = out[:, 2]
    return hcat(cov_hat, cov_sum_hat)
end

# Given a parameter combination, find the model-implied moments.
function compute_moments(params)
    σ_ζ, σ_ϵ = params

    # Moments of income
    # cov = ρ * σ_ζ / (1 - ρ^2)
    # var = σ_ζ / (1 - ρ^2) + σ_ϵ

    # Moments of the change in income (differenced moments)
    # var_d = 2 * (σ_ζ / (1 + ρ)) + 2 * σ_ϵ
    # cov_d = -1 * (1 - ρ) * (σ_ζ / (1 + ρ)) - σ_ϵ
    # Note: var + 2 * cov is an equation of only σ_ζ, so this will work.

    # Third try: using the conditions from the slides 
    
    cov_d_tilde = -ρ * σ_ϵ
    cov_sum_d_tilde = ρ * σ_ζ

    # Return as a vector of moments
    return [cov_d_tilde, cov_sum_d_tilde]
end

# Compute the GMM objective value given parameters and sample moments.
function objective(params::Vector{Float64}, W::Matrix{Float64}, sample_moments::Vector{Float64})
    model_moments = compute_moments(params)
    g_hat = model_moments .- sample_moments
    obj_value = g_hat' * W * g_hat
    return obj_value
end

# Once the optimizing parameters are found, compute the GMM variance-covariance matrix estimate.
function compute_gmm_variance(params::Vector{Float64}, W::Matrix{Float64})
    S = W # Just identified case

    D = ForwardDiff.jacobian(compute_moments, params)

    var_matrix = inv(D' * W * D) * (D' * W * S * W * D) * inv(D' * W * D)
    return var_matrix
end

# Compute second-step weighting matrix using the sample moments.
function compute_W(params::Vector{Float64}, sample_moments)
    n = size(sample_moments, 1)
    k = size(sample_moments, 2)
    g = zeros(n, k)
    model_mom = compute_moments(params)
    for i = 1:n
        g[i, :] = sample_moments[i, :] .- model_mom
    end

    # Compute new weighting matrix from residual covariance matrix.
    W_out = inv((1 / n) * (g' * g))
    return W_out
end

# Estimates the model parameters using GMM.
function GMM(df::DataFrame, W::Matrix{Float64},ρ::Float64)
    N = size(df, 1) # Consider each ixt a different observation

    sample_moments = hcat(est_moments(df,ρ))
    pooled_sample_moments = mean(sample_moments, dims=1)[1, :]

    # Initial guess for the parameters.
    σ_init = ones(2)

    # First-step: minimize the objective function.
    result = optimize(σ -> objective(σ, W, pooled_sample_moments), σ_init, NelderMead(), Optim.Options(f_tol=1e-8))
    σ_optimal = Optim.minimizer(result)

    #= 
    # Second-step (currently commented out)
    W_2 = compute_W(σ_optimal, sample_moments)
    result_2ndstep = optimize(σ -> objective(σ, W_2, pooled_sample_moments), σ_optimal, NelderMead(), Optim.Options(f_tol=1e-8))
    σ_optimal_2nd_step = Optim.minimizer(result_2ndstep)
    =#

    # Compute the variance-covariance matrix.
    var_matrix = compute_gmm_variance(σ_optimal, W)

    # Compute the standard errors (square root of diagonal elements divided by N).
    σ_SE = sqrt.(diag(var_matrix)./N)
    
    return σ_optimal, σ_SE
end

############################################################
# Compute the variance of persistent and transitory income: 
############################################################
W = Matrix{Float64}(I, 2, 2)

σ_optimal, σ_SE = GMM(df, W, ρ)
############################################################
# Output
############################################################
cd(outdir)

results_df = DataFrame(
    Parameter = ["σ_ζ", "σ_ϵ",], 
    Estimate = σ_optimal,
    StdError = σ_SE,
)

CSV.write("estimation_results.csv", results_df)

println(results_df)