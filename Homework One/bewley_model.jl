###########################################
# Estimates the Bewley model 
# using income as a sum of a deterministic process, 
# a persistent (AR1) income process and a transitory process.  
###########################################
# Packages 
###########################################
using Parameters, CSV, DelimitedFiles, CSV, Plots,Distributions,LaTeXStrings,Statistics, DataFrames
###########################################
# Put your path to the dataset "estimation_results.csv" below 
indir = "$outdir_parameters" 

# Put the outdirectory below
outdir_images = "$outdir_images" 
outdir_parameters = "$outdir_parameters" 
###########################################
# Parameters
###########################################
cd(indir)
@with_kw struct Model_Parameters @deftype Float64

    # Load discretized income processes estimated in "discretize_income_process.jl" 

    # Grids
    ϵ_grid::Array{Float64,1}  = readdlm("transitory_grid.csv")[:,1]
    ζ_grid::Array{Float64,1}  = readdlm("permanent_grid.csv")[:,1]

    σ_0_grid::Array{Float64,1}  = readdlm("initial_permanent_dist.csv")[:,1]

    # Transition Matrices
    T_ϵ::Matrix{Float64} = readdlm("transitory_T.csv")
    T_ζ::Matrix{Float64} = readdlm("permanent_T.csv")

    nϵ::Int64 = length(ϵ_grid)
    nζ::Int64 = length(ζ_grid)

    # Load the lifecycle component 
    κ::Matrix{Float64} = hcat(CSV.File("life_cycle_income.csv").age,CSV.File("life_cycle_income.csv").deterministic_component)

    B = 0.01 # No-borrowing constraint
    a_min = B
    a_max = 1000000 # Maximum assets allowed
    na::Int64 = 500 # Number of capital grid points 

    # As the function will be most concave at a = 0 
    # Create an exponentially-spaced grid, like in Kaplan and Violante (2010)
    a_grid::Array{Float64,1} = exp.(collect(range(log(a_min), length = na, stop = log(a_max)))) 

    r = 0.04 # Assumed interest rate 

    σ = 2.0 # Coefficient of Relative Risk Aversion 
    β = 0.975 # Discount rate 
    N::Int64 = 35 # Years of Life
    ρ = 0.97
end 

#initialize value function and policy functions
mutable struct Solutions

    val_func::Array{Float64,4}
    pol_func::Array{Float64,4}

end

function build_solutions(para) 

    val_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )
    pol_func = zeros(Float64,para.na,para.nϵ,para.nζ,para.N )

    sols = Solutions(val_func,pol_func)

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
                                start_index = index_ap
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
plot(age_grid, sqrt.(var(consumption,dims = 1)[1,:]),xlabel = "Age", ylabel = L"Standard Deviation ($)",label = "",title = "Consumption Inequality by Age")
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