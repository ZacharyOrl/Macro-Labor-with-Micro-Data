################################################################
# Econ 810: Spring 2025 Advanced Macroeconomics 
# Discretizes a continuous income process with i.i.d shocks
# uses the Rouwenhorst method 
################################################################
# Add Packages 
using Plots, Statistics, Optim, LinearAlgebra, DataFrames, CSV, DelimitedFiles, Distributions, LaTeXStrings
################################################################
# Put your path to the dataset "estimation_results.csv" below 
indir = "$outdir_parameters" 

# Put the two outdirectories below
outdir_params = indir
outdir_images = "$outdir_images" 
cd(indir)
################################################################
# Parameters 
ρ = 0.97 # correlation in the persistent component of income 
nζ = 20 # Grid points for the permanent component
nϵ = 10 # Grid points for the transitory component

W = Matrix{Float64}(I, 3, 3) # Use the identity matrix as we are in the just-identified case. 

# Load parameters estimated in "est_var_params.jl" 
params = CSV.File("estimation_results.csv")
σ_ζ = params.Estimate[1] 
σ_ϵ = params.Estimate[2] 
################################################################
# GMM Functions for Rouwenhorst
################################################################

# Given a parameter combination, find the implied moments 
# of the discretized distribution 
# using that the transition probabilities for high N 
# can be approximated using a binomial. 

function compute_moments(params::Vector{Float64},N::Int64)
# [moment vector] = compute_moments(candidate parameter vector, chosen number of grid points)
    p,q,ψ = params

    s = (1 - p) / (2 - (p + q))

    # Compute moments
    expec = (q - p) * ψ / (2 - (p + q))
    corr = p + q - 1
    var = ψ^2 * (1 - 4s*(1-s) + (4s * (1-s))/(N-1))
    
    # Return as a vector
    return [expec,corr,var]
end

# Compute the GMM objective value given some parameters and the sample moments. 
function objective(params::Vector{Float64}, W::Matrix{Float64}, sample_moments::Vector{Float64},N::Int64)
    model_moments = compute_moments(params,N)
    g_hat = model_moments .- sample_moments
    obj_value = g_hat' * W * g_hat

    return obj_value
end 

function GMM(σ::Float64,ρ::Float64,N::Int64,W::Matrix{Float64})
    sample_moments = [0 ρ σ/(1-ρ^2)][1,:]

    # Initial guess for the parameters
    params_init = 1/2 * ones(3)

    # Minimize the objective function
    result = optimize(p -> objective(p, W, sample_moments,N), params_init, BFGS())

    # Extract the optimized parameters
    param_optimal = Optim.minimizer(result)
    
    return param_optimal
end
############################################################
# Find the transition matrix & grid points
############################################################

function compute_transition_matrix(p_hat::Vector{Float64},N::Int64)
    p = p_hat[1]
    q = p_hat[2]

    T = [p 1-p ; 1-q q]

    for i = 2:N-1
        zero_vec = zeros(i,1)
        T = p * [T zero_vec; zero_vec' 0] + (1-p) * [zero_vec T; 0 zero_vec'] + 
            (1-q) * [zero_vec' 0 ; T zero_vec] + q * [0 zero_vec' ; zero_vec T] 

    # Divide all but the top and bottom two rows by 2 so that their elements sum to one
        for n = 2:size(T,2) - 1
            T[n,:] = 1/2 * T[n,:]
        end
        
    end 

    return T 
end 

function generate_grid(p_hat::Vector{Float64},N::Int64)
    ψ = p_hat[3]
    y_N = ψ
    y_1 = -y_N

    grid = collect(range(start=y_1, stop=ψ, length=N))

    return grid 
end 

function simulate_distributions(S::Int64)
    # Find the stationary distribution of discretized transitory and persistent income processes 
    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ϵ[1,:])

    # Statte-contingent distributions over the permanent components
    perm_dists = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Need to choose some initial state to start the markov-chain 
    initial_dist = perm_dists[5]

    persistent = zeros(S)
    transitory = zeros(S)

    index_transitory = rand(transitory_dist)
    index_persistent = rand(initial_dist)

    # Persistent and Transitory components 
    persistent[1] = ζ_grid[index_persistent]
    transitory[1] = ϵ_grid[index_transitory]
    for s = 2:S

        index_persistent = rand(perm_dists[index_persistent]) # Draw the new permanent component based upon the old one. 
        index_transitory = rand(transitory_dist) # Draw the transitory component 
        

        # Outputs 
        persistent[s] = ζ_grid[index_persistent]
        transitory[s] = ϵ_grid[index_transitory]

    end 

    return persistent,transitory 
end 

############################################################
# Discretize the initial distribution of the permanent 
# component. 
############################################################
function compute_initial_dist(σ_0::Float64,grid::Vector{Float64})
    n = length(grid) 

    dist =zeros(n)
    dist[1] = cdf(Normal(0, σ_0), grid[1])

    for i = 2:n-1
        dist[i] = cdf(Normal(0, σ_0), grid[i]) - cdf(Normal(0, σ_0), grid[i - 1])
    end 

    dist[n] = 1 - cdf(Normal(0, σ_0), grid[n - 1])

    return dist
end

############################################################
# Compute the process
############################################################

# Compute the parameters for each process
p_hat_ζ = GMM(σ_ζ,ρ,nζ,W)
p_hat_ϵ = GMM(σ_ϵ,0.0,nϵ,W)

T_ζ = compute_transition_matrix(p_hat_ζ,nζ)
ζ_grid = generate_grid(p_hat_ζ,nζ)

T_ϵ = compute_transition_matrix(p_hat_ϵ,nϵ)
ϵ_grid = generate_grid(p_hat_ϵ,nϵ)

# Compute the initial distribution of permanent income 
σ_0 = sqrt(0.15) # Taken from Kaplan and Violante (2010)

σ_0_ζ_dist = compute_initial_dist(σ_0,ζ_grid)

############################################################
# Plots of distributions
############################################################
cd(outdir_images)

S = 20000 
persistent,transitory = simulate_distributions(S)

# Remove some burn-in for the initial state
start = 1000

histogram(persistent[start:S], xlabel = L"\tilde{P}_{i,t}",label ="",normalize = :probability)
savefig("PS1_Image_04.png") 
histogram(transitory[start:S], xlabel = L"\tilde{\epsilon}_{i,t}",label ="",normalize = :probability)
savefig("PS1_Image_05.png") 
############################################################
# Output
############################################################
cd(outdir_params)

writedlm("transitory_grid.csv", ϵ_grid, ',')
writedlm("permanent_grid.csv", ζ_grid, ',')

writedlm("transitory_T.csv", T_ϵ, ' ')
writedlm("permanent_T.csv", T_ζ, ' ')

writedlm("initial_permanent_dist.csv", σ_0_ζ_dist, ',')