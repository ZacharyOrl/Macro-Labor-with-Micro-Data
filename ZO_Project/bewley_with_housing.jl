###########################################
# Estimates the Bewley model with Housing
# using income as a sum of a deterministic process, 
# a persistent (AR1) income process and a transitory process.  
# housing follows a deterministic trend. 
###########################################
# Packages 
###########################################
using Parameters, CSV, DelimitedFiles, CSV, Plots,Distributions,LaTeXStrings,Statistics, DataFrames, LinearAlgebra, Optim
###########################################
# Put your path to the dataset "estimation_results.csv" below 
indir = "C:/Users/zacha/Documents/2025 Spring/Advanced Macroeconomics/J. Carter Braxton/Homework/ZO_Project/"

# Put the outdirectory below
outdir_images = joinpath(indir, "images")
outdir_parameters = joinpath(indir, "parameters")

###########################################
# Functions
###########################################
cd(indir)
# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

# Function which computes an initial distribution over a given grid 
include("compute_initial_dist.jl")

###########################################
# Parameters
###########################################
cd(joinpath(indir, "parameters"))

@with_kw struct Model_Parameters @deftype Float64

    # Housing parameters
    b::Float64 = 0.016 # Annual Log House Price increase used in Cocco (2005) - he estimates 0.016 but uses a lower number to account for quality improvement. 
    d::Float64 = 0.15                               # Limit on Home equity you can take out in any period. 
    #π::Float64 = compound(1 + 0.032, 5) - 1        # Moving shock probability 
    #δ::Float64 = compound(1 + 0.01, 5) - 1         # Housing Depreciation
    λ::Float64 = 0.08                               # House-sale cost 
    γ::Float64 = 0.2  # Down-payment proportion 

    # Rent cost: 
    # In 2000 it was $602 a month and home values were $119,600 so about renters spend about 6% of the typical home value renting 
    # In 1960, the typical house price was 58,600 (in 2000 dollars)
    # Source: https://www2.census.gov/programs-surveys/decennial/tables/time-series/coh-values/values-adj.txt
    rent_prop::Float64 = 0.06 
    P_bar::Float64 = 58600 #About 2 x median household income in 1960 https://kinder.rice.edu/urbanedge/can-texas-afford-lose-its-housing-affordability-advantage

    # Utility function parameters
    # θ::Float64 = 0.1              # Utility from housing services relative to consumption
    σ = 2.0 # Coefficient of Relative Risk Aversion 
    β = 0.975 # Discount rate 
    
    r = 0.04 # Assumed interest rate 
    
    N::Int64 = 35 # Years of Life

    # Variance parameters 
    # Same as in Week 1
    σ_ζ::Float64 = 0.02285
    σ_ϵ::Float64 = 0.03863

    # Correlation parameters
    # Same as in Week 1
    φ = 0.97 # Autocorrelation in the persistent component of income.

    # Load discretized income processes estimated in "discretize_income_process.jl" 

    # Persistent Component 
    ζ_grid::Vector{Float64} = rouwenhorst(σ_ζ/(1 - φ^2), φ, 5)[1]
    T_ζ::Matrix{Float64} = rouwenhorst(σ_ζ/(1 - φ^2), φ, 5)[2]

    nζ::Int64 = length(ζ_grid)

    # Initial Persistent Component 
    σ_0_grid::Array{Float64,1}  = compute_initial_dist(0.0, sqrt(0.15), η_grid)

    # Transitory Component 
    ϵ_grid::Vector{Float64} = rouwenhorst(σ_ϵ, 0.0, 5)[1]
    T_ϵ::Matrix{Float64} = rouwenhorst(σ_ϵ, 0.0, 5)[2]

    nϵ::Int64 = length(ϵ_grid)

    # Load the lifecycle component of income
    κ::Matrix{Float64} = hcat(CSV.File("life_cycle_income.csv").age,CSV.File("life_cycle_income.csv").deterministic_component)
    
    # Cash-on-Hand
    X_min = -1000000.0
    X_max = -1 * X_min # Maximum cash on hand allowed
    nX::Int64 = 100 # Number of cash on hand grid points 
    X_grid::Array{Float64,1} = collect(range(X_min, length = nX, stop = X_max))

    # Mortgage Debt
    # Not interested in studying the impact of house price risk on consumption 
    # So House Prices are deterministic. 
    M_min = 0.0
    M_max = (1-λ) * P_bar * exp(b * (N-1))
    nM::Int64 = 80 # Number of mortgage debt grid points 

    M_grid::Array{Float64,1} = collect(range(M_min, length = nM, stop = M_max))
end 

#initialize value function and policy functions
mutable struct Solutions

    val_func::Array{Float64,4}
    H_pol_func::Array{Float64,4}
    X_pol_func::Array{Float64,4}
    M_pol_func::Array{Float64,4}

end

function build_solutions(para) 

    val_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )
    H_pol_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )
    X_pol_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )
    M_pol_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )

    sols = Solutions(val_func,H_pol_func,X_pol_func, M_pol_func)

    return sols
end 

function Initialize_Model() 

    para = Model_Parameters()
    sols = build_solutions(para)

    return para, sols 
end
#########################################################
# Functions 
#########################################################
function Solve_Problem(para::Model_Parameters, sols::Solutions)
    # Solves the decision problem, outputs results back to the sols structure. 

    @unpack na, nϵ, nζ, N, a_grid, ϵ_grid, ζ_grid, r, σ, β, T_ϵ, T_ζ, κ = para
    @unpack val_func, pol_func = sols

    V_next = zeros(na, nϵ, nζ, N + 1) 
    pol_next = zeros(na, nϵ, nζ, N + 1)

    println("Begin solving the model backwards")
    for j in N:-1:1  # Backward induction
        println("Age is ", 24+j)
        for e in 1:nϵ
            ϵ = ϵ_grid[e]

            for z in 1:nζ
                ζ = ζ_grid[z]
                candidate_max = -Inf                     
                Y = exp(ϵ + ζ + κ[j,2])  # Construct the income process

                # Use that ap(a) is a weakly increasing function. 
                start_index = 1 
                for index_a in 1:na
                    a = a_grid[index_a]
                    coh =  (1 + r) * a + Y
                    for index_ap in start_index:na 
                        ap = a_grid[index_ap]
                        c = coh - ap 

                        if c > 0  # Feasibility check
                            val = (c^(1 - σ)) / (1 - σ)

                            for e_prime in 1:nϵ
                                for z_prime in 1:nζ
                                    val += β * T_ϵ[e,e_prime] * T_ζ[z,z_prime] * V_next[index_ap, e_prime, z_prime, j + 1]
                                end
                            end

                            if val > candidate_max  # Check for max
                                candidate_max = val
                                pol_next[index_a, e, z, j] = ap
                               # start_index = index_ap
                                V_next[index_a, e, z, j] = candidate_max
                            end   
                        end
                    end 
                end 
            end 
        end
    end

 
    sols.val_func .= V_next[:,:,:,1:N]
    sols.pol_func .= pol_next[:,:,:,1:N]
end

function simulate_model(para,sols,S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 

    @unpack na, nϵ, nζ, N, a_grid, ϵ_grid, ζ_grid, r, σ, β, T_ϵ, T_ζ, κ,σ_0_grid = para
    @unpack val_func, pol_func = sols

    # Distribution over the initial permanent component
    initial_dist = Categorical(σ_0_grid)

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ϵ[1,:])

    # State-contingent distributions over the permanent components
    perm_dists = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Outputs
    assets = zeros(S,N) # Saving by Age
    consumption = zeros(S,N) 
    persistent = zeros(S,N)
    transitory = zeros(S,N)
    income = zeros(S,N) 

    for s = 1:S
        index_transitory = rand(transitory_dist)
        index_persistent = rand(initial_dist)

        index_a = 1 # Start with 0 assets

        # Asset policy 
        assets[s,1] = pol_func[index_a,index_transitory,index_persistent,1]

        # Persistent and Transitory components 
        persistent[s,1] = ζ_grid[index_persistent]
        transitory[s,1] = ϵ_grid[index_transitory]

        # Compute income
        income[s,1] = exp(persistent[s,1] + transitory[s,1] + κ[1,2]) 

        # Consumption policy 
        consumption[s,1] = a_grid[index_a]*(1+r) + income[s,1] - assets[s,1]

        for n = 2:35 
            index_persistent = rand(perm_dists[index_persistent]) # Draw the new permanent component based upon the old one. 
            index_transitory = rand(transitory_dist) # Draw the transitory component 
            index_a = findfirst(x -> x == assets[s,n-1], a_grid) # Find the index of the previous choice of ap 

            # Outputs 
            assets[s,n] = pol_func[index_a,index_transitory,index_persistent,n]
            persistent[s,n] = ζ_grid[index_persistent]
            transitory[s,n] = ϵ_grid[index_transitory]
            income[s,n] = exp(persistent[s,n] + transitory[s,n] + κ[n,2])
            consumption[s,n] = a_grid[index_a]*(1+r) + income[s,n] - assets[s,n]
        end 
    end 

    return assets, consumption,persistent,transitory, income
end

##########################################
# Estimate insurance pass-through coefficients
##########################################

# Using the BPP method: 
# Pass through of transitory shocks 

function BPP_insurance_est(para,sols,transitory::Matrix{Float64},persistent::Matrix{Float64},consumption::Matrix{Float64},ρ::Float64)
    S = size(income,1)
    N = size(income,2)
    y = transitory .+ persistent
    log_consumption = log.(consumption)

    consumption_growth = diff(log_consumption,dims = 2)
    # Construct tilde(y_growth) = y today - ρ * y yesterday)
    y_growth = zeros(S,N-1)
    for s = 1:S
        for i = 2:N
            y_growth[s,i-1] = y[s,i] - ρ * y[s,i-1]
        end 
    end 
    
    y_growth_sum = ρ^2 .*  y_growth[:,1:N-3] .+ ρ .*  y_growth[:,2:N-2] .+  y_growth[:,3:N-1] 
    # Compute pass-throughs 

    a = zeros(S)
    b = zeros(S)
    for s = 1:S
        a[s] = cov(consumption_growth[s,1:N-2],y_growth[s,2:N-1])
        b[s] = cov(y_growth[s,1:N-2],y_growth[s,2:N-1])
    end 

    α_ϵ = 1 - sum(a) / sum(b)

    c = zeros(S)
    d = zeros(S)
    for s = 1:S
        c[s] = cov(consumption_growth[s,2:N-2],y_growth_sum[s,:])
        d[s] = cov(y_growth[s,2:N-2],y_growth_sum[s,:])
    end 

    α_ζ = 1- sum(c) / sum(d)
    return α_ϵ, α_ζ
end 

function true_insurance(sols,transitory::Matrix{Float64},persistent::Matrix{Float64},consumption::Matrix{Float64},ρ::Float64)
    S = size(income,1)
    N = size(income,2)

    log_consumption = log.(consumption)
    consumption_growth = diff(log_consumption,dims = 2)

    a = zeros(S)
    b = zeros(S)
    for s = 1:S
        a[s] = cov(consumption_growth[s,1:N-1],transitory[s,2:N])
        b[s] = var(transitory[s,1:N-1])
    end 

    α_ϵ = 1 - sum(a) / sum(b)

    ζ =  persistent[:,2:N] .- ρ .* persistent[:,1:N-1]
    c = zeros(S)
    d = zeros(S)
    for s = 1:S
        c[s] = cov(consumption_growth[s,1:N-1],ζ[s,:])
        d[s] = var(ζ[s,:])
    end 

    α_ζ = 1- sum(c) / sum(d)
    return α_ϵ,α_ζ
end
###################################################
# Solve the model and simulate 
###################################################
para,sols = Initialize_Model()
Solve_Problem(para,sols)
assets, consumption,persistent, transitory, income = simulate_model(para,sols,20000)

###################################################
# Plots 
###################################################
cd(outdir_images)

start_age = 25 
end_age = 59

age_grid = collect(range(start_age, length = end_age - start_age + 1, stop = end_age))

# Plot wealth statistics by age 
plot(age_grid, mean(assets,dims = 1)[1,:], label = "Mean")
plot!(age_grid, median(assets,dims = 1)[1,:],label = "Median",xlabel = "Age", ylabel = L"Wealth ($)", title = "Wealth Accumulation over the Lifecycle")
savefig("PS1_Image_01.png") 

# Plot consumption variance by age 
plot(age_grid, (var(log.(consumption),dims = 1)[1,:]),xlabel = "Age", ylabel = "Variance of logs",label = "",title = "Consumption Inequality by Age")
savefig("PS1_Image_02.png") 

# Plot the histogram of assets for the age with the highest average wealth - to ensure the upper bound is reasonable
# for this starting value. 
histogram(assets[:,20],xlabel = L"Wealth ($)",label ="", title = "Distribution of Wealth at Age 45",normalize = :probability)
savefig("PS1_Image_03.png") 
######################################
# Compute insurance coefficients
######################################
α_ϵ_hat,α_ζ_hat =   BPP_insurance_est(para,sols,transitory,persistent,consumption,para.ρ)
α_ϵ,α_ζ = true_insurance(sols,transitory,persistent,consumption, para.ρ)

############################################################
# Output
############################################################
cd(outdir_parameters)

results_df = DataFrame(
    Parameter = ["α_ϵ", "α_ζ",], 
    BPP = [α_ϵ_hat,α_ζ_hat],
    True = [α_ϵ,α_ζ],
)

CSV.write("insurance_coefficients.csv", results_df)

println(results_df)