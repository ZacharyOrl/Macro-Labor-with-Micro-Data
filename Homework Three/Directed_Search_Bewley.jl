#= ################################################################################################## 
    Econ 810: Spring 2025 Advanced Macroeconomics 
    Authors:    Zachary Orlando and Cutberto Frias Sarraf
=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, DataFrames,FastGaussQuadrature
using CategoricalArrays, StatsPlots

# Directory where the images will be stored
outdir = "C:/Users/zacha/Documents/2025 Spring/Advanced Macroeconomics/J. Carter Braxton/Homework/Homework Three/images"
#= ################################################################################################## 
    Part 2: Model
=# ##################################################################################################
cd(outdir)
#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives @deftype Float64

    T::Int64 = 120                           # Life-cycle to 30 years (quarterly)
    r        = (1.04)^(1/4) − 1              # Quarterly net interest rate corresponding to an annualized 4% rate. 
    β        = 0.99                          # Discount rate  
    σ        = 2.0                           # Coefficient of Relative Risk Aversion 
    δ        = 0.1                           # Job-Destruction Rate

    τ        = 0.03                           # Marginal tax rate on wages
    ζ        = 1.6                           # Matching elasticity parameter
    κ        = .995                          # Cost of posting a vacancy for a firm

    p_hl     = 0.5                           # Per-period probability of moving down one human capital grid point when unemployed
    p_hh     = 0.05                          # Per-period probability of moving up one human capital grid point when employed

    # Grids
    # Human Capital
    h_min          = 1.0
    h_max          = 2.0
    nh::Int64      = 25
    h_grid::Vector{Float64} = range(h_min, h_max, length=nh)   

    # Piece Wages 
    w_min           = 0.0 # w > z/(h_min *(1-τ)
    w_max           = 1.0
    nw::Int64       = 25
    w_grid::Vector{Float64} = range(w_min, w_max, length=nw)   

    # Saving  - use b to match assignment notation
    b_min = 0.01  # (Default Calibration assumes a ZBC)
    b_max = 10.0
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

    z::Float64                      # Unemployment Benefit (Transfer funded by τ)
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

    z = 0.4
    results  = Results(U, W, J, W_policy, U_w_policy, U_b_policy, θ, z)
    return param, results
end

#= ################################################################################################## 

    Functions

=# ##################################################################################################
function flow_utility_func(c::Float64, param::Primitives)
    @unpack σ, = param

    return ( c^( 1 - σ ) - 1   ) / (1 - σ)
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
# Being unemployed is better than being employed if z > (1-τ)wh
                        if c <= 0
                            continue
                        end

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

                        if c <= 0
                            continue
                        end

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


    # Loop over each individual 
    for s = 1:S 
        # Every person starts their life unemployed, with 0 savings and the lowest human capital. 
        saving[s,1]        = b_grid[1] 
        saving[s,2]        = U_b_policy[1, 1, 1]
        human_capital[s,1] = h_grid[1] 
        w_search[s,1]      = U_w_policy[1, 1, 1]
        employed[s,1]      = 0 
        taxes[s,1]         = 0.0

        consumption[s,1]   = z + saving[s,1] .- (1/(1 + r)) .* saving[s,2]
        # Loop over each lifecycle 
        for t = 2:T
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
                        saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index]
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
                            saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index - 1]
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
                            saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index]
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
                        saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index]
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
                            saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index + 1]
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
                            saving[s, t + 1] = U_b_policy[t, saving_index, human_capital_index]

                            consumption[s, t] =  z + saving[s,t] - (1/(1 + r)) * saving[s, t + 1]
                        end 
                    end 
                end 
            end 
        end 
    end 

    return w_search,  employed, human_capital, consumption, saving,  taxes
end 
##########################################################################
# Solve and simulate the model 
##########################################################################
param, results = Initialize_Model()

Solve_Problem(param, results)

S = 10000
w_search,  employed, human_capital, consumption, saving,  taxes = simulate_model(S, param, results)
##########################################################################
# Plots 
##########################################################################
T = 120 
age_grid = range(25.0, 55, length=T)  

# a) Histogram of the distribution of assets

# Plot the histogram of savings at age 45 (T = 80)
histogram(saving[:,80],ylabel = "Fraction of Simulations",label ="", title =  "Distribution of assets at age 45",normalize = :pdf )
savefig("PS3_Image_01.png") 

# b) Histogram of wages

# Plot the distribution of wages conditional on actually receiving the wage 
masked_wages = ifelse.(employed .== 1, w_search, missing)
histogram(masked_wages[:,80],ylabel = "Fraction of Simulations",label ="", title =  "Distribution of wages at age 45",normalize = :pdf )
savefig("PS3_Image_02.png") 

# c) Unemployment rate 
# Don't include the first year when everyone is trivially unemployed. 
plot(age_grid[2:120], 1 .- mean(employed[:,2:120], dims = 1)', ylabel = "Unemployment Rate",label = "", xlabel = "Age", title = "Unemployment Rate by Age")
savefig("PS3_Image_03.png") 

# Unemployment Rate overall is: 
1 .- mean(employed[:,2:120]) # 13.8%
# d) Mean earnings and assets by age 
# Construct earnings from tax payments: earnings = taxes * ((1 - τ)/τ) (I.e. exclude income from savings)
earnings = taxes .* ((1 - param.τ)/param.τ)

# Plot average & median (because there is likely skewness) earnings by age 
plot(age_grid[2:120], mean(earnings[:,2:120], dims = 1)', ylabel = "After-Tax Earnings",label = "Mean", xlabel = "Age", title = "Earnings by Age")
plot!(age_grid[2:120], median(earnings[:,2:120], dims = 1)', ylabel = "After-Tax Earnings",label = "Median", xlabel = "Age", title = "Earnings by Age")
savefig("PS3_Image_04.png") 

# Plot mean & median savings by age 
plot(age_grid[2:120], mean(saving[:,2:120], dims = 1)', ylabel = "Savings",label = "Mean", xlabel = "Age", title = "Assets by Age")
plot!(age_grid[2:120], median(saving[:,2:120], dims = 1)', ylabel = "",label = "Median", xlabel = "Age", title = "Assets by Age")
savefig("PS3_Image_05.png") 

# e) Find the average gain in earnings while employed

# Find the earnings individuals received when they were employed only 
masked_earnings = ifelse.(employed .== 1, earnings, missing)
earnings_change = masked_earnings[:,2:T] .- masked_earnings[:, 1:T-1]

# The mean earnings change is: 
mean(skipmissing(earnings_change)) # 0.0017

# Which is about 0.2% of earnings for the employed - much lower than the earnings growth of about 3% we found in the PSID.  
100 * mean(skipmissing(earnings_change))/mean(skipmissing(masked_earnings))

# f) Find individuals around job loss and plot their earnings changes: 
employment_changes     = employed[:, 2:T] .- employed[:, 1:T-1] 
job_loss  = employment_changes .== -1 

function compute_event_study(outcome, event, employment_condition)
    # Outputs average values of the outcome 4 periods before to 8 periods after each event
    # Job loss goes from t = 2 to T
    S, T = size(outcome)
    event_window = -3:9  # From t-4 to t+8
    results = zeros(length(event_window))
    event_count = 0
    
    for s in 1:S, t in 4:T-9
        if event[s, t] == 1 && employment_condition[s, t] == 1 && employment_condition[s, t - 1] == 1 && employment_condition[s, t - 2] == 1 && employment_condition[s, t - 3] == 1
            for (i, τ) in enumerate(event_window)
                results[i] += (outcome[s, t + τ] -  outcome[s,t-3])/ outcome[s, t - 3]
            end
            event_count += 1
        end
    end
    
    return event_count > 0 ? results ./ event_count : results
end

estimates_earnings = compute_event_study(earnings,job_loss, employed)

# Make a graph of of earnings losses around job_loss - normalizing earnings to an initial year. 
plot(-4:8, estimates_earnings, xlabel = "Quarters relative to job-loss", ylabel = "Earnings Loss\nRelative to Initial Earnings",label="", title = "Effect of Layoff on Earnings")
savefig("PS3_Image_06.png") 

# g) Make a graph of consumption around job loss
estimates_consumption = compute_event_study(consumption,job_loss, employed)

plot(-4:8,estimates_consumption,xlabel = "Quarters relative to job-loss", ylabel = "Consumption Fall \nRelative to Initial Consumption",label="", title = "Effect of Layoff on Consumption")
savefig("PS3_Image_07.png") 
##################################################################################
# h) Increase z by 10% and re-do the plots for earnings and consumption. What changes? 
##################################################################################
results.z = 0.44 

Solve_Problem(param, results)

# Simulate new model
S = 10000
w_search_z_cf,  employed_z_cf, human_capital_z_cf, consumption_z_cf, saving_z_cf,  taxes_z_cf, = simulate_model(S, param, results)

# Record employment changes
employment_changes_z_cf     = employed_z_cf[:, 2:T] .- employed_z_cf[:, 1:T-1] 
job_loss_z_cf = employment_changes_z_cf .== -1 

# Estimate earnings 
earnings_z_cf = taxes_z_cf .* ((1 - param.τ)/param.τ)

# Make new earnings and consumption estimates 
estimates_earnings_z_cf = compute_event_study(earnings_z_cf,job_loss_z_cf, employed_z_cf)

estimates_consumption_z_cf = compute_event_study(consumption_z_cf,job_loss_z_cf, employed_z_cf)

# Earnings graph
plot(-4:8, estimates_earnings, xlabel = "Quarters relative to job-loss", ylabel = "Earnings Loss\nRelative to Initial Earnings",label="z = 0.4", title = "Effect of Layoff on Earnings")
plot!(-4:8, estimates_earnings_z_cf,label = "z = 0.44")
savefig("PS3_Image_08.png")

# Consumption graph
plot(-4:8, estimates_consumption, xlabel = "Quarters relative to job-loss", ylabel = "Consumption Loss\nRelative to Initial Consumption",label="z = 0.4", title = "Effect of Layoff on Consumption")
plot!(-4:8, estimates_consumption_z_cf,label = "z = 0.44")
savefig("PS3_Image_09.png")

# Find the unemployment rate 
plot(age_grid[2:120], 1 .- mean(employed[:,2:120], dims = 1)', ylabel = "Unemployment Rate",label = "z = 0.4", xlabel = "Age", title = "Unemployment Rate")
plot!(age_grid[2:120], 1 .- mean(employed_z_cf[:,2:120], dims = 1)',label = "z = 0.44")
savefig("PS3_Image_10.png")

# Unemployment Rate overall is: 
1 .- mean(employed_z_cf[:,2:120]) # 13.8%




