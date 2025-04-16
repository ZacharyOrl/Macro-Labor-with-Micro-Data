#= ################################################################################################## 
    Econ 810: Spring 2025 Advanced Macroeconomics 
    Authors:    Zachary Orlando and Cutberto Frias Sarraf
=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, DataFrames,FastGaussQuadrature
using CategoricalArrays, StatsPlots

#= ################################################################################################## 
    Part 2: Model
=# ##################################################################################################

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives @deftype Float64

    T::Int64 = 120                           # Life-cycle to 30 years (quarterly)
    r        = (1.04)^(1/4) − 1              # Quarterly net interest rate corresponding to an annualized 4% rate. 
    β        = 0.99                          # Discount rate  
    σ        = 2.0                           # Coefficient of Relative Risk Aversion 
    δ        = 0.1                           # Job-Destruction Rate

    τ        = 0.2                           # Marginal tax rate on wages
    ζ        = 1.6                           # Matching elasticity parameter
    κ        = .995                          # Cost of posting a vacancy for a firm
    z        = 0.4                           # Unemployment Benefit (Transfer funded by τ)

    p_hl     = 0.5                           # Per-period probability of moving down one human capital grid point when unemployed
    p_hh     = 0.05                          # Per-period probability of moving up one human capital grid point when employed

    # Grids
    # Human Capital
    h_min          = 0.5
    h_max          = 1.5
    nh::Int64      = 25
    h_grid::Vector{Float64} = range(h_min, h_max, length=nh)   

    # Piece Wages 
    w_min           = 0.0
    w_max           = 1.0
    nw::Int64       = 25
    w_grid::Vector{Float64} = range(w_min, w_max, length=nw)   

    # Saving  - use b to match assignment notation
    b_min = 0.01  # (Default Calibration assumes a ZBC)
    b_max = 5.0
    nb::Int64      = 100
    b_grid::Vector{Float64} = range(b_min, b_max, length=nb)

end 

#initialize value function and policy functions
@with_kw mutable struct Results
    U::Array{Float64,3}             # U[T, b, h]
    W::Array{Float64,4}             # W[T, w, b, h]
    J::Array{Float64,3}             # J[T, w, h]

    W_policy::Array{Float64,4}      # Once matched, the only choice is b -  pol[T, w, b, h]
    U_w_policy::Array{Float64,3}    # Need to choose a w to search in and a b 
    U_b_policy::Array{Float64,3}    # Need to choose a w to search in and a b

    θ::Array{Float64,3}             # Market Tightness of each submarket for each period θ[T, w, h]
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    U          = zeros(T + 1, nb, nh)
    W          = zeros(T + 1, nw, nb, nh) 
    J          = zeros(T + 1, nw, nh) 

    W_policy   = zeros(T, nw, nb, nh)
    U_w_policy   = zeros(T, nb, nh)       
    U_b_policy   = zeros(T, nb, nh) 

    θ          = zeros(T+1, nw, nh)

    results  = Results(U, W, J, W_policy, U_w_policy, U_b_policy, θ)
    return param, results
end

#= ################################################################################################## 

    Functions

=# ##################################################################################################
function flow_utility_func(c::Float64, param::Primitives)
    @unpack σ, = param

    return ( c^( 1 - σ )   ) / (1 - σ)
end 

function iterate_firm_value(j::Int64, param::Primitives, results::Results)
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

    # Solve the firm's problem for each h and w, saving the θ and J 
    for h_index in 1:nh
        h = h_grid[h_index]
        
        if h_index == nh
            for w_index in 1:nw
                w = w_grid[w_index]

                # Firm's value function.  
                J[j, w_index, h_index] = (1-w)*h + β*(1-δ) * J[j+1, w_index, h_index]     

                # Find the market tightness from the inverting the free entry condition: 

                # Impose non-negativity on tightness
                if (J[j, w_index, h_index]/κ)^ζ - 1 < 0 
                    θ[j, w_index, h_index] = 0.0 
                
                else 
                    θ[j, w_index, h_index] = ((J[j, w_index, h_index]/κ)^ζ - 1)^(1/ζ) 
                end    
                                        
            end
        else 
            for w_index in 1:nw
                w = w_grid[w_index]     
                
                # Firm's value function.  
                J[j, w_index, h_index] = (1-w) * h + β * ( (1-δ) *  (1 - p_hh) *  J[j+1, w_index, h_index] + (1-δ) *  p_hh *  J[j+1, w_index, h_index + 1] )    

                # Find the market tightness from the inverting the free entry condition: 
                if (J[j, w_index, h_index]/κ)^ζ - 1 < 0 
                    θ[j, w_index, h_index] = 0.0 
                
                else 
                    θ[j, w_index, h_index] = ((J[j, w_index, h_index]/κ)^ζ - 1)^(1/ζ) 
                end              
            end
        end
    end
end 


function iterate_employee_value(j::Int64, param::Primitives, results::Results)
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

      # Loop over the borrowing/saving states and choices for individuals 
      for b_index in 1:nb
        b = b_grid[b_index]

        # Solve the employed worker's problem conditional on a borrowing state: 
        # Loop of employed worker's human capital states
        for h_index in 1:nh
            h = h_grid[h_index]
            
            # If at the highest human capital state, human capital cannot move upwards. 
            if h_index == nh
                for w_index in 1:nw
                    w = w_grid[w_index]
                    
                    # Find the saving choice of an employed worker
                    candidate_max = -Inf     
                    
                    for b_prime_index in 1:nb
                        b_prime = b_grid[b_prime_index]
                        
                        # Use the employed's budget constraint to find their consumption 
                        c = b - (1/(1+ r)) * b_prime + (1-τ) * w * h 

                        if c > 0  # Feasibility check
                            val = flow_utility_func(c, param) 

                            val += β * ((1 -δ) * W[j+1, w_index, b_prime_index, h_index] + δ * U[j+1, b_prime_index, h_index])

                            if val > candidate_max 
                                candidate_max = val 

                                W[j, w_index, b_index, h_index] = val 
                                W_policy[j, w_index, b_index, h_index] = b_prime

                            end 

                        end 
                    end 
                                            
                end
            else 
                for w_index in 1:nw
                    w = w_grid[w_index]  
                    
                    # Find the saving choice of an employed worker
                    candidate_max = -Inf     
                    
                    for b_prime_index in 1:nb
                        b_prime = b_grid[b_prime_index]
                        
                        # Use the employed's budget constraint to find their consumption 
                        c = b - (1/(1+ r)) * b_prime + (1-τ) * w * h 

                        if c <= 0
                            continue
                        end

                        if c > 0  # Feasibility check
                            val = flow_utility_func(c, param)

                            val += β * (      p_hh * ( (1-δ) * W[j+1, w_index, b_prime_index, h_index + 1] + δ * (U[j+1, b_prime_index, h_index + 1])) + 
                                        (1 - p_hh) * ( (1-δ) * W[j+1, w_index, b_prime_index, h_index]     + δ * (U[j+1, b_prime_index, h_index]) ) 
                                        )
                            
                            if val > candidate_max 
                                candidate_max = val 

                                W[j, w_index, b_index, h_index] = val 
                                W_policy[j, w_index, b_index, h_index] = b_prime
                            end 
                        end 
                    end 
                end
            end
        end 
    end 
end 

function iterate_unemployed_value(j::Int64, param::Primitives, results::Results)
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

    # Loop over the borrowing/saving states and choices for individuals 
      for b_index in 1:nb
        b = b_grid[b_index]

        for h_index in 1:nh
            h = h_grid[h_index]

            # Find the w and b choice of an unemployed worker
            if h_index == 1
                candidate_max = -Inf
                
                # w choice
                for w_index in 1:nw
                    w = w_grid[w_index] 
                    
                    # Find the probability of finding a match in the submarket from market tightness 
                    p = 1/((1/θ[j+1, w_index, h_index])^ζ + 1)^(1/ζ) 

                    # b_choice
                    for b_prime_index in 1:nb
                        b_prime = b_grid[b_prime_index]

                        c = z + b - (1/(1 + r)) * b_prime

                        if c > 0  # Feasibility check
                            val = flow_utility_func(c, param) 
                            
                            val += β * (    p * W[ j+1, w_index, b_prime_index, h_index] + 
                                        (1-p) * U[ j+1, b_prime_index, h_index] )

                            if val > candidate_max
                                candidate_max = val

                                U[ j, b_index, h_index] = val
                                U_w_policy[ j, b_index, h_index] = w
                                U_b_policy[ j, b_index, h_index] = b_prime
                            end 
                        end 
                    end 
                end 

                 # Find the w and b choice of an unemployed worker
            else
                candidate_max = -Inf
                
                # w choice
                for w_index in 1:nw
                    w = w_grid[w_index] 
                    
                    # Find the probability of finding a match in the submarket from market tightness 
                    p_fall = 1/((1/θ[j+1, w_index, h_index - 1])^ζ + 1)^(1/ζ) 

                    p_stay = 1/((1/θ[j+1, w_index, h_index])^ζ + 1)^(1/ζ) 

                    # b_choice
                    for b_prime_index in 1:nb
                        b_prime = b_grid[b_prime_index]

                        c = z + b - (1/(1 + r)) * b_prime

                        if c <= 0
                            continue
                        end

                        if c > 0  # Feasibility check
                            val = flow_utility_func(c , param) 
                            
                            val += β * ( p_hl       * ( p_fall * W[ j+1, w_index, b_prime_index, h_index - 1] + (1-p_fall) * U[ j+1, b_prime_index, h_index - 1] ) + 
                                         (1 - p_hl) * ( p_stay * W[ j+1, w_index, b_prime_index, h_index]     + (1-p_stay) * U[ j+1, b_prime_index, h_index] ) 
                                        )     

                            if val > candidate_max
                                candidate_max = val

                                U[ j, b_index, h_index] = val
                                U_w_policy[ j, b_index, h_index] = w
                                U_b_policy[ j, b_index, h_index] = b_prime
                            end 

                        end 

                    end

                end 
            end 
        end 
    end 
end 

function Solve_Problem(param::Primitives, results::Results)
    # Solves the decision problem, outputs results back to the sols structure. 
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

    println("Begin solving the model backwards")
    for j in T:-1:1  # Backward induction
        println("Age is ", 25 + (j-1)/4)

       iterate_firm_value(j, param, results)
       iterate_employee_value(j, param, results)
       iterate_unemployed_value(j, param, results)
    end 

end 

##########################################################################
# Simulate the model 
##########################################################################
function simulate_model(S::Int64, param::Primitives, results::Results)
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

    # Outputs
    w_search = zeros(S,T) # Search by Age - if the agent is employed they have search value of their current employment. 
    employed = zeros(S,T) # 0 or 1 indicating whether the individual was employed in that quarter
    human_capital = zeros(S,T) # the human capital level of the agent 
    consumption = zeros(S,T) # the consumption of the agent 
    saving = zeros(S,T + 1) # the savings of the agent
    taxes = zeros(S,T)

    # Every person starts their life unemployed, with 0 savings and the lowest human capital. 
    saving[:,1]        .= b_grid[1] 
    saving[:,2]        .= U_b_policy[1, 1, 1]
    human_capital[:,1] .= h_grid[1] 
    w_search[:,1]      .= U_w_policy[1, 1, 1]
    employed[:,1]      .= 0 
    taxes[:,1]         .= 0 

    consumption[:,1]   .= z .+ saving[:,1] .- (1/(1 + r)) .* saving[:,2]


    # Loop over each lifecycle 
    for t = 2:T
            # Loop over each individual 
        for s = 1:S 
            w_search_index      = findfirst(x -> x == w_search[s,t-1], w_grid)
            human_capital_index = findfirst(x -> x == human_capital[s,t-1], h_grid)
            saving_index        = findfirst(x -> x == saving[s,t], b_grid)

            # Consider the unemployed: 
            if employed[s,t-1] == 0 

                # If their human capital is at the boundary, they will have the same human capital tomorrow 
                if human_capital_index == 1
                    human_capital[s, t] = human_capital[s, t - 1]

                    # As h is constant, their probability of employment tomorrow is then a single value: 
                    p_employed = 1/((1/θ[t, w_search_index, human_capital_index])^ζ + 1)^(1/ζ) 

                    # They become employed with probability p_employed 
                    if rand() < p_employed 
                        employed[s,t] = 1 

                        w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                        saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index]

                        consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                        taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]
                    
                    # If they do not become employed
                    else 
                        employed[s, t] = 0
                        w_search[s, t] = U_w_policy[t, saving_index, human_capital_index]
                        saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index]
                        consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]

                    end 
                # If their human capital is not at the lower boundary
                else 
                    # They lose a human capital level with probability p_hl
                    if rand() < p_hl
                        human_capital[s, t] = h_grid[human_capital_index - 1]

                        # As h is constant, their probability of employment tomorrow is then a single value: 
                        p_employed = 1/((1/θ[t, w_search_index, human_capital_index - 1])^ζ + 1)^(1/ζ) 

                        # They become employed with probability p_employed 
                        if rand() < p_employed 
                            employed[s,t] = 1 

                            w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                            saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index - 1]

                            consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                            taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]
                        
                        # If they do not become employed
                        else 
                            employed[s, t] = 0
                            w_search[s, t] = U_w_policy[t, saving_index, human_capital_index - 1]
                            saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index - 1]
                            consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]

                        end 
                    # They do not lose a human capital level
                    else 
                        human_capital[s, t] = h_grid[human_capital_index]

                        # As h is constant, their probability of employment tomorrow is then a single value: 
                        p_employed = 1/((1/θ[t, w_search_index, human_capital_index])^ζ + 1)^(1/ζ) 

                        # They become employed with probability p_employed 
                        if rand() < p_employed 
                            employed[s,t] = 1 

                            w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                            saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index]

                            consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                            taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]
                        
                        # If they do not become employed
                        else 
                            employed[s, t] = 0
                            w_search[s, t] = U_w_policy[t, saving_index, human_capital_index]
                            saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index]
                            consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]

                        end 
                    end 
                end 
            # If they are employed 
            else 
                # If their human capital index is at the upper boundary, they will not move 
                if human_capital_index == nh 
                    human_capital[s, t] = human_capital[s, t - 1]

                    # They remain employed with probability 1-δ
                    if rand() < (1-δ)
                        employed[s,t] = 1 

                        w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                        saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index]

                        consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                        taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]

                    # They become unemployed with probability δ
                    else 
                        employed[s, t] = 0
                        w_search[s, t] = U_w_policy[t, saving_index, human_capital_index]
                        saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index]
                        consumption[s, t] =  z + saving[s, t] - (1/(1 + r)) * saving[s, t + 1]
                    end 
                # They are not at the upper boundary of the human capital grid 
                else                     
                    # They gain a human capital level with probability p_hh
                    if rand() < p_hh 

                        human_capital[s, t] = h_grid[human_capital_index + 1]
                        # Remain employed 
                        if rand() < (1-δ)
                            employed[s,t] = 1 

                            w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                            saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index + 1]

                            consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                            taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]

                        # Become unemployed  
                        else 
                            employed[s, t] = 0
                            w_search[s, t] = U_w_policy[t, saving_index, human_capital_index + 1]
                            saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index + 1]
                            consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]
                        end 
                    # They do not gain a human capital level 
                    else 
                        human_capital[s, t] = h_grid[human_capital_index]
                        # Remain employed 
                        if rand() < (1-δ)
                            employed[s,t] = 1 

                            w_search[s, t] = w_search[s, t-1] # Record their search today as whatever they searched yesterday. 
                            saving[s, t + 1] = W_policy[t, w_search_index, saving_index, human_capital_index]

                            consumption[s, t] = saving[s ,t] - (1/(1+ r)) * saving[s, t + 1] + (1-τ) * w_search[s, t] * human_capital[s, t] 
                            taxes[s, t] = τ * w_search[s, t] * human_capital[s, t]

                        # Become unemployed  
                        else 
                            employed[s, t] = 0

                            w_search[s, t] = U_w_policy[t, saving_index, human_capital_index]
                            saving[s, t + 1] = U_b_policy[t, w_search_index, saving_index, human_capital_index]

                            consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]
                        end 
                    end 
                end 
            end 
        end 
    end 

    return w_search,  employed, human_capital, consumption, saving,  taxes
end 

w_search,  employed, human_capital, consumption, saving,  taxes = simulate_model(2000, param, results)

mean(consumption, dims = 1)
plot(consumption)
##########################################################################
# Checks
##########################################################################
param, results = Initialize_Model()

Solve_Problem(param, results)
_Problem(param, results)
# Check the results for tightness and firm utiltiy 
@unpack_Primitives param                # Unpack model parameters
@unpack_Results results

plot(results.U[1,40, :])


J[T,1,25] # 1.5 as expected
J[T-1,1,25] 
plot(results.J[:,1, 25]) # Falls with t

θ[T,:,:]
θ[2,:,:] # Tightness falls over time as expected. 

W_policy[50,:,41,:] # Saving policy seems ok 
U_b_policy[1,1,:]
U[100,2,:]

U_w_policy[1,1,:]

U[T,:,:]




