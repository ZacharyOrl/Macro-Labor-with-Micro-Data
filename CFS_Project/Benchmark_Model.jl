#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Final Project: Unemployment risk in a Life-cycle Bewely economy

    Last Edit:  April 17, 2025
    Authors:    Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, LaTeXStrings, Distributions

include("Tauchen_Hussey_1991.jl")

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives

    T::Int64        = 35                            # Life-cycle to 35 years
    r::Float64      = 0.04                          # Interest rate  
    β::Float64      = 0.975                         # Discount rate  
    γ::Float64      = 2.0                           # Coefficient of Relative Risk Aversion 

    # Assets Grid
    a_min::Float64  = 0.1                           # Minimum value of assets
    a_max::Float64  = 100000.0                       # Maximum value of assets
    na::Int64       = 100                           # Grid points for assets
    a_grid::Vector{Float64} = exp.(collect(range(log(a_min), length = na, stop = log(a_max))))   

    # Income process
    ρ::Float64      = 0.97                          # Correlation in the persistent component of income 
    nζ::Int64       = 11                            # Grid points for the permanent component
    σ_ζ::Float64    = sqrt(0.01)                    # Standard deviation of the permanent shock

    nϵ::Int64       = 5                             # Grid points for the transitory component
    σ_ϵ::Float64    = sqrt(0.05)                    # Standard deviation of the transitory shock

    κ::Vector{Float64} = [                          # Deterministic age profile for log-income
    10.00571682417030, 10.06468173213630, 10.14963371320800, 10.18916005760660, 10.25289993933830,
    10.27787916956560, 10.32260755975800, 10.36733797632800, 10.39391841908670, 10.42305441774350,
    10.45397023113620, 10.48282124181770, 10.50757066459240, 10.53338513735820, 10.53669397036220,
    10.56330457698600, 10.58945748446780, 10.60438525029320, 10.62570875544670, 10.62540348055990,
    10.63732032711820, 10.64422326499790, 10.65124153265610, 10.64869889614020, 10.60836674391850,
    10.61725912807620, 10.60201099108720, 10.58990416581600, 10.55571432462690, 10.54753392025080,
    10.53038700787840, 10.51112990486990, 10.50177243660240, 10.49346004128460, 10.48778926452950
    ]

end 

# Initialize value function and policy functions
@with_kw mutable struct Results
    V::Array{Float64,4}
    a_policy::Array{Float64,4}
    a_policy_index::Array{Int64,4}
    c_policy::Array{Float64,4}
end

@with_kw struct OtherPrimitives
    ζ_grid::Vector{Float64}
    T_ζ::Matrix{Float64}
    ϵ_grid::Vector{Float64}
    T_ϵ::Vector{Float64}
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    V                   = zeros(T + 1, na, nζ, nϵ)          # Value function
    a_policy            = zeros(T, na, nζ, nϵ)              # Savings function
    a_policy_index      = zeros(T, na, nζ, nϵ)
    c_policy            = zeros(T, na, nζ, nϵ)              # Consumption function

    ζ_grid, T_ζ         = tauchen_hussey(nζ, ρ  , σ_ζ)      # Discretization of Permanent shocks  [ζ]
    ϵ_grid, T_ϵ         = tauchen_hussey(nϵ, 0.0, σ_ϵ)      # Discretization of Transitory shocks [ϵ]
    T_ϵ                 = T_ϵ[1,:]
    
    other_param         = OtherPrimitives(ζ_grid, T_ζ, ϵ_grid, T_ϵ)
    results             = Results(V, a_policy, a_policy_index, c_policy)

    return param, results, other_param
end

#= ################################################################################################## 
  
    Functions

=# ##################################################################################################

# Flow_Utility(300000.0, param)
# Flow_Utility(600000.0, param)

function Flow_Utility(c::Float64, param::Primitives)
    @unpack_Primitives param                
    return (c^(1 - γ)) / (1 - γ)
end 

function Solve_Problem(param::Primitives, results::Results, other_param::OtherPrimitives)
    # Solves the decision problem: param is the structure of parameters and results stores solutions 
    @unpack_Primitives param                
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    println("Begin solving the model backwards")
    for j in T:-1:1  # Backward induction
        println("Age is ", 24+j)
        κ_j = κ[j]

    #= --------------------------------- STATE VARIABLES ----------------------------------------- =#
        for ζ_index in 1:nζ                                             # State: Permanent shock ζ 
                ζ = ζ_grid[ζ_index]
                
            for ϵ_index in 1:nϵ                                     # State: Transitory shock ϵ
                ϵ = ϵ_grid[ϵ_index]
                Y = exp(κ_j + ζ + ϵ)                                # Income in levels 

                # start_index = 1                                     # Use that a'(a) is a weakly increasing function. 
                @inbounds for a_index in 1:na                       # State: Assets a
                    a = a_grid[a_index]
                    X = Y + (1 + r) * a

                    candidate_max = -Inf
    #= --------------------------------- DECISION VARIABLES -------------------------------------- =#
                    @inbounds for ap_index in 1:na    # Control: Assets a': 1:na start_index:na
                    ap = a_grid[ap_index]
                    c  = X - ap                                 # Consumption
    #= --------------------------------- GRID SEARCH --------------------------------------------- =#
                        if c <= 0                               # Feasibility check
                            continue
                        end

                        # Compute expected value
                        EV = 0.0
                        @inbounds for ζp_index in 1:nζ
                            for ϵp_index in 1:nϵ
                                EV += T_ζ[ζ_index, ζp_index] * T_ϵ[ϵp_index] * V[j+1, ap_index, ζp_index, ϵp_index]
                            end
                        end

                        val = Flow_Utility(c, param) + β * EV  # Utility Value

                        if val > candidate_max                  # Check for max
                            candidate_max                                   = val
                            a_policy[j, a_index, ζ_index, ϵ_index]          = ap
                            c_policy[j, a_index, ζ_index, ϵ_index]          = c
                            a_policy_index[j, a_index, ζ_index, ϵ_index]    = ap_index
                            # start_index                                     = ap_index
                            V[j, a_index, ζ_index, ϵ_index]                 = val
                        end  

                    end 
                end 
            end 
        end

    end
end

#= ################################################################################################## 
    Solving the Model
=# ##################################################################################################
param, results, other_param = Initialize_Model()
Solve_Problem(param, results, other_param)
@unpack_Primitives param                                             
@unpack_Results results
@unpack_OtherPrimitives other_param

#= ################################################################################################## 
    Simulations
=# ##################################################################################################

function simulate_model(param, results, other_param, S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    @unpack_Primitives param                                             
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    # Distribution over the initial permanent component
    ζ0_grid, T_ζ0  = tauchen_hussey(nζ, 0.0, 0.15)        # Discretization of Initial Permanent shocks  [ζ]    
    Initial_dist   = Categorical(T_ζ0[1,:])

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    Transitory_dist = Categorical(T_ϵ)

    # State-contingent distributions over the permanent components
    Perm_dists      = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Outputs
    Assets          = zeros(S,N) # Saving by Age
    Consumption     = zeros(S,N) 
    Persistent      = zeros(S,N)
    Transitory      = zeros(S,N)
    Income          = zeros(S,N) 

    for s = 1:S
        index_transitory = rand(Transitory_dist)
        index_persistent = rand(Initial_dist)

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

#= ################################################################################################## 
    Plots
=# ##################################################################################################

age_grid = 25:54

# Value Policy Function: Assets
age       = [25, 35, 45, 54]
indices   = [ 1, 11, 21, 30]
plot(a_grid/1000, V[indices[1], :, 6, 3], label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, V[idx, :, 6, 3], label = "Age = $t")
end
title!("")
xlabel!("Assets a (in thousand)")
ylabel!("Value function")
plot!(legend=:bottomright)

# Savings Policy Function: Assets
age       = [25, 35, 45, 54]
indices   = [ 1, 11, 21, 30]
plot(a_grid/1000, a_policy[indices[1], :, 6, 3]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, a_policy[idx, :, 6, 3]/1000, label = "Age = $t")
end
title!("")
xlabel!("Assets a (in thousand)")
ylabel!("Assets a' (in thousands)")
plot!(legend=:bottomright)

# Consumption Policy Function: Assets
age       = [25, 35, 45, 54]
indices   = [ 1, 11, 21, 30]
plot(a_grid/1000, c_policy[indices[1], :, 11, 5]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, c_policy[idx, :, 11, 5]/1000, label = "Age = $t")
end
title!("")
xlabel!("Assets a (in thousand)")
ylabel!("Consumption c (in thousands)")
plot!(legend=:bottomright)



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