#= ################################################################################################## 
    Econ 810: Spring 2025 Advanced Macroeconomics 
    Authors:    Zachary Orlando and Cutberto Frias Sarraf
=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, DataFrames,FastGaussQuadrature
using CategoricalArrays, StatsPlots

#= ################################################################################################## 
    Part 2: Model
    Consider a simplified life-cycle version of the model in Ljungqvist and Sargent (1998). Suppose 
    workers have linear utility (risk neutral) and live for T periods. To find jobs, workers exert 
    search intensity s at utility cost c(s). Given that effort, π(s) is the probability of 
    receiving a job offer. Job offers are drawn from stationary distribution F(w). When employed, 
    suppose that each period there is a probability δ of being laid off. Let h denote human capital, 
    and take home pay will be wh. When unemployed, workers receive a transfer b, which is common to 
    all workers.
=# ##################################################################################################

include("Tauchen_1986.jl")

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives

    T::Int64    = 360                           # Life-cycle to 30 years (monthly)
    r::Float64  = 0.04                          # Interest rate  
    β::Float64  = (1 / (1 + r))^ (1 / 12)       # Discount rate  
    δ::Float64  = 0.033                         # Probability of being laid off
    b::Float64  = 0.10                          # Value of unemployment insurance
    
    # Grids
    h_min::Float64 = 1.0
    h_max::Float64 = 2.0
    nh::Int64      = 25
    h_grid::Vector{Float64} = range(h_min, h_max, length=nh)   

    s_min::Float64 = 0.0
    s_max::Float64 = 1.0
    ns::Int64      = 41
    s_grid::Vector{Float64} = range(s_min, s_max, length=ns)

    c::Vector{Float64}  = 0.5 .* s_grid                 # Search cost
    PI::Vector{Float64} = sqrt.(s_grid)                 # Probability of drawing and offer

    μ::Float64   = 0.50
    σ::Float64   = 0.10 # sqrt(0.10)
    nw::Int64    = 41
end 

#initialize value function and policy functions
@with_kw mutable struct Results
    U::Array{Float64,2}
    W::Array{Float64,3}
    S_policy::Array{Float64,2}
    S_policy_index::Array{Int64,2}
    W_policy::Array{Float64,2}
    w_reservation::Array{Float64,2}
    ψᵤ::Float64 
    ψₑ::Float64
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    U          = zeros(T + 1, nh)
    W          = zeros(T + 1, nh, nw) 
    S_policy   = zeros(T, nh)
    S_policy_index = zeros(T, nh)
    W_policy   = zeros(T, nh)           
    w_reservation = zeros(T, nh)                 # Reservation wage
    ψᵤ         = 0.50
    ψₑ         = 0.05 # 0.20

    w_grid, w_prob = tauchen(nw, 0.0, σ, μ)      # Wage offer distribution 
    w_prob         = w_prob[1,:]

    results  = Results(U, W, S_policy, S_policy_index,W_policy, w_reservation, ψᵤ, ψₑ)
    return param, results, w_grid, w_prob
end

#= ################################################################################################## 
    2.2 Assignment
     
    Solve the model with VFI and simulate a mass of agents.
    In the VFI there are two policy functions to store: (1) search policy function and (2) reservation 
    wage by human capital. Each policy function is also a function of age.
          
    Functions

=# ##################################################################################################

function Solve_Problem(param::Primitives, results::Results)
    # Solves the decision problem, outputs results back to the sols structure. 
    @unpack_Primitives param                # Unpack model parameters
    @unpack_Results results

    println("Begin solving the model backwards")
    for j in T:-1:1  # Backward induction
        println("Age is ", 25 + (j-1)/12)
        for h_index in 1:nh
            h = h_grid[h_index]
                  
            # Value function for the employed
            if h_index == nh
                for w_index in 1:nw
                    w = w_grid[w_index]       
                    W[j, h_index, w_index] = w * h + β * ((1 -δ) * W[j+1, h_index, w_index] + δ * U[j+1, h_index])                             
                end
            else 
                for w_index in 1:nw
                    w = w_grid[w_index]       
                    W[j, h_index, w_index] = w * h + β * ψₑ * ((1 -δ) * W[j+1, h_index+1, w_index] + δ * U[j+1, h_index+1]) + 
                                                 β * (1-ψₑ) * ((1 -δ) * W[j+1, h_index  , w_index] + δ * U[j+1, h_index  ])                               
                end
            end

            candidate_max = -Inf                     
            # Value function for the Unemployed
            for s_index in 1:ns
                s = s_grid[s_index]
                utility = b - c[s_index]

                val = utility
                if h_index == 1
                    for w_index in 1:nw
                        val += β * PI[s_index] * w_prob[w_index] * max(W[j+1, h_index, w_index], U[j+1, h_index]) 
                    end
                        val += β * (1-PI[s_index]) * U[j+1, h_index]                              
                else
                    for w_index in 1:nw
                        val += β * ψᵤ     * PI[s_index] * w_prob[w_index] * max(W[j+1, h_index-1, w_index], U[j+1, h_index-1]) + 
                               β * (1-ψᵤ) * PI[s_index] * w_prob[w_index] * max(W[j+1, h_index  , w_index], U[j+1, h_index  ])                              
                    end
                        val += β * ψᵤ * (1-PI[s_index]) * U[j+1, h_index-1] + β * (1-ψᵤ) * (1-PI[s_index]) * U[j+1, h_index]   
                end 

                if val > candidate_max          # Check for max
                    candidate_max               = val
                    S_policy[j, h_index]        = s
                    S_policy_index[j, h_index]  = s_index
                    U[j, h_index]               = candidate_max
                end   
            end     
        end
    end

    for j in 1:T
        for h_index in 1:nh
            # println("j=$j h_index=$h_index")
            for w_index in 1:nw
                if W[j, h_index, w_index] >= U[j, h_index]
                    w_reservation[j, h_index] = w_grid[w_index]
                    # println("w=$(w_grid[w_index]), W=$(W[j, h_index, w_index]), U=$(U[j, h_index])")
                    break  # First w_index satisfying the condition is the reservation wage
                end
            end
        end 
    end


end

#= ################################################################################################## 
    Solving the Model
=# ##################################################################################################
param, results, w_grid, w_prob = Initialize_Model()
Solve_Problem(param, results)
@unpack_Primitives param                                             
@unpack_Results results

#= ################################################################################################## 
    Plots
=# ##################################################################################################

# Search Policy Function
age = [25, 30, 35, 40, 45, 50, 55]
indices = [1, 60, 120, 180, 240, 300, 360]
plot(h_grid, S_policy[indices[1], :], label = "t = $(age[1])", ylims = (0.8, 1))
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(h_grid, S_policy[idx, :], label = "t = $t", ylims = (0.8, 1))
end
title!("Search Policy Function")
xlabel!("Human Capital")
ylabel!("Search Policy Function")
plot!(legend=:bottomright)
# savefig("Homework Two/Output/PS2_Image_01.png") 

# Reservation Wage 
# age = [25, 30, 35, 40, 45, 50, 55]
# indices = [1, 60, 120, 180, 240, 300, 360]
# plot(h_grid, w_reservation[indices[1], :], label = "t = $(age[1])")
# for (t, idx) in zip(age[2:end], indices[2:end])
#     plot!(h_grid, w_reservation[idx, :], label = "t = $t")
# end
# title!("Reservation Wage")
# xlabel!("Human Capital")
# ylabel!("Reservation Wage")
# plot!(legend=:bottomleft)
# savefig("Homework Two/Output/PS2_Image_02.png") 

# Reservation Wage 
human_capital = [1.0, 1.25, 1.50, 2.0]
indices = [1, 7, 13, 25]
plot(1:360, w_reservation[:, indices[1]], label = "Human Capital = $(human_capital[1])")
for (t, idx) in zip(human_capital[2:end], indices[2:end])
    plot!(1:360, w_reservation[:, idx], label = "Human Capital = $t")
end
title!("Reservation Wage")
xlabel!("Age in Months")
ylabel!("Reservation Wage")
plot!(legend=:bottomleft)
# savefig("Homework Two/Output/PS2_Image_02.png") 

#= ################################################################################################## 
    Additional Plots
=# ##################################################################################################

# Value Function Unemployment
age = [25, 30, 35, 40, 45, 50, 55]
indices = [1, 60, 120, 180, 240, 300, 360]
plot(h_grid, U[indices[1], :], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(h_grid, U[idx, :], label = "t = $t")
end
title!("Value Function Unemployment")
xlabel!("Human Capital")
ylabel!("Value Function Unemployment")
plot!(legend=:topleft)
# savefig("Homework Two/Output/PS2_Image_03.png") 

# Value Function Employment
age = [25, 30, 35, 40, 45, 50, 55]
indices = [1, 60, 120, 180, 240, 300, 360]
plot(h_grid, W[indices[1], :, 41], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(h_grid, W[idx, :, 41], label = "t = $t")
end
title!("Value Function Employment")
xlabel!("Human Capital")
ylabel!("Value Function Employment")
plot!(legend=:topleft)
# savefig("Homework Two/Output/PS2_Image_04.png") 

#= ################################################################################################## 
    In the simulated data, plot the distribution of human capital among the employed and
    unemployed. Do the distribution look like you would expect?   
=# ##################################################################################################

function simulate_model(param, results, S::Int64)

    @unpack_Primitives param                                             
    @unpack_Results results 

    # Initial distirbution of human capital 
    h_indices          = rand(1:length(h_grid), S) # Random indices
    Initial_h_Dist     = h_grid[h_indices]         # Actual values from the grid

    # Distrbution of wages
    W_dist             = Categorical(w_prob)

    # Outputs
    Human_Capital       = zeros(S, T) 
    Human_Capital_Index = zeros(Int64, S, T)
    Employment          = zeros(Int64, S, T)
    Wage                = zeros(S, T) 
    Search_Intensity    = zeros(S, T) 
    Search_Index        = zeros(Int64, S, T)

    for s = 1:S

        # Initial Human Capital
        Human_Capital_Index[s,1] = h_indices[s]
        Human_Capital[s,1]       = Initial_h_Dist[s]

        # Initial Search Policy 
        Search_Intensity[s,1] = S_policy[1, h_indices[s]]
        Search_Index[s,1]     = S_policy_index[1, h_indices[s]]

        for n = 2:T 

            # Unemployment
            if Employment[s, n-1] == 0 

                # Human capital
                if rand() > ψᵤ
                    Human_Capital_Index[s, n] = Human_Capital_Index[s, n-1]
                    Human_Capital[s, n]       = Human_Capital[s, n-1]
                else
                    Index_Decrease = max(Human_Capital_Index[s, n-1] - 1, 1)
                    Human_Capital_Index[s, n] = Index_Decrease
                    Human_Capital[s, n]       = h_grid[Index_Decrease]
                end

                # Receive a Job Offer
                Offer              = rand() < PI[Search_Index[s, n-1]]

                if Offer == 1
                    W_drawn_index    = rand(W_dist)   # Draw a job offer

                    if w_grid[W_drawn_index] >= w_reservation[n, Human_Capital_Index[s, n]]
                        Employment[s, n] = 1
                        Wage[s, n]       = w_grid[W_drawn_index]
                    else
                        Employment[s, n] = 0
                        Wage[s, n]       = b
                        Search_Intensity[s,n] = S_policy[n, Human_Capital_Index[s,n]]
                        Search_Index[s,n]     = S_policy_index[n, Human_Capital_Index[s,n]]
                    end
                else 
                    Employment[s, n] = 0
                    Wage[s, n]       = b
                    Search_Intensity[s,n] = S_policy[n, Human_Capital_Index[s,n]]
                    Search_Index[s,n]     = S_policy_index[n, Human_Capital_Index[s,n]]
                end 

            # Employment    
            else
                # Human capital
                if rand() > ψₑ
                    Human_Capital_Index[s, n] = Human_Capital_Index[s, n-1]
                    Human_Capital[s, n]       = Human_Capital[s, n-1]
                else
                    Index_Increase = min(Human_Capital_Index[s, n-1] + 1, nh)
                    Human_Capital_Index[s, n] = Index_Increase
                    Human_Capital[s, n]       = h_grid[Index_Increase]
                end

                # Probability of being laid off
                if rand() > δ
                    Employment[s, n] = Employment[s, n-1]
                    Wage[s, n]       = Wage[s, n-1]
                else
                    Employment[s, n] = 0
                    Wage[s, n]       = b
                    Search_Intensity[s,n] = S_policy[n, Human_Capital_Index[s,n]]
                    Search_Index[s,n]     = S_policy_index[n, Human_Capital_Index[s,n]]
                end

            end 

        end 

    end 

    return Human_Capital, Employment, Wage, Search_Intensity
end

S = 2000
Human_Capital, Employment, Wage, Seach_Intensity = simulate_model(param, results, S)

#= ################################################################################################## 
    In the simulated data, plot the distribution of human capital among the employed and unemployed. 
    Do the distribution look like you would expect?
=# ##################################################################################################

Human_Capital_Vector  = vec(Human_Capital)
Employment_Vector     = vec(Employment)

# Conditional Values
Human_Capital_Employed   = Human_Capital_Vector[Employment_Vector .== 1]
Human_Capital_Unemployed = Human_Capital_Vector[Employment_Vector .== 0]

histogram([Human_Capital_Employed, Human_Capital_Unemployed],
          label  = ["Employed" "Unemployed"],
          title  = "Historgram of Human Capital Observations",
          xlabel = "Human Capital",
          bins   = 48, 
          legend =:topleft,
          alpha  = 0.6)
# savefig("Homework Two/Output/PS2_Image_05.png") 

#= ################################################################################################## 
    In the simulated data, what is the average gain in earnings for individuals who are working for 
    two consecutive years (i.e., 24 periods without a δ shock)? How does this estimate compare to 
    your estimate from the PSID data in Section 1? How does this estimate vary as you increase/
    decrease the parameter ψe.
=# ##################################################################################################

Income_changes = Float64[]
for i in 1:S
    # Find all 24-month employment spells
    for t in 1:(360 - 24 + 1)  # Loop over time periods where a 24-month spell could start
        # Check if the individual is employed for 24 months
        if all(Employment[i, t:t+23] .== 1)
            initial_income = sum(Wage[i, t:t+11] .* Human_Capital[i, t:t+11])
            final_income   = sum(Wage[i, t+12:t+23] .* Human_Capital[i, t+12:t+23])
            # Calculate income change
            income_change = final_income - initial_income
            push!(Income_changes, income_change)
        end
    end
end
Mean_income_change = mean(Income_changes)
Std_income_change  = std(Income_changes)
println("Mean Income Change: ", Mean_income_change)
println("Standard Deviation of Income Change: ", Std_income_change)

#= ################################################################################################## 
    ψu = 0.50 and ψe = 0.05
    Mean Income Change:                  0.1421
    Standard Deviation of Income Change: 0.1673    

    ψu = 0.50 and ψe = 0.20
    Mean Income Change:                  0.1750
    Standard Deviation of Income Change: 0.3035
=# ##################################################################################################

#= ################################################################################################## 
    In the simulated data, plot the average path of earnings from 6 months before to 2 years after 
    an unemployment spells (i.e., a δ shock in your model). What is the percent decline in earnings 
    after 2-years? How does this compare to the data reported in Davis and von Wachter (2011) or 
    Jarosch (2014)? What happens if you increase/decrease ψu.
=# ##################################################################################################

before_months   = 6
after_months    = 24
income_dynamics = []

for i in 1:S # Individuals
    for t in 7:T  # Find indices where the individual lost their job
        if Employment[i, t] == 0 && all(Employment[i, t-6:t-1] .== 1) # First time losing job (at least in the last 6 months)

            # Obtain range for months before and after the event
            start_month = max(1, t - before_months)
            end_month   = min(T, t + after_months)
            
            # Calculate income            
            wages         = Wage[i, start_month:end_month]
            human_capital = Human_Capital[i, start_month:end_month]
            income        = wages .* human_capital
            
            # Make sure it has the correct length
            months_before = t - start_month  # Number of months before the event
            months_after  = end_month - t    # Number of months after the event
            
            if length(income) < (before_months + after_months + 1)
                # We need to ensure uniform length (before + after + event month)
                fill_size_before = before_months - months_before
                fill_size_after  = after_months - months_after

                # Fill with zeros if the vector is too short
                fill_income = vcat(fill(missing, fill_size_before), income, fill(missing, fill_size_after))
                push!(income_dynamics, convert(Vector{Union{Missing, Float64}}, fill_income))
            else
                push!(income_dynamics, convert(Vector{Union{Missing, Float64}}, income))
            end
        end
    end
end

# Calculate the average dynamics across all individuals and events
max_len                = maximum(map(length, income_dynamics))
Income_dynamics        = [vcat(income, fill(missing, max_len - length(income))) for income in income_dynamics]
Income_dynamics_matrix = hcat(Income_dynamics...)
transposed_matrix      = transpose(Income_dynamics_matrix)
Avg_income_dynamics    = [mean(skipmissing(transposed_matrix[:, i])) for i in 1:size(transposed_matrix, 2)]
Avg_income_dynamics    = vec(Avg_income_dynamics)
months_range           = -before_months:after_months

Avg_income_dynamics_Psi_u_050 = Avg_income_dynamics
Avg_income_dynamics_Psi_u_025 = Avg_income_dynamics
Avg_income_dynamics_Psi_u_005 = Avg_income_dynamics

# This plot was made with ψu = 0.50 and ψe = 0.20
plot(months_range, Avg_income_dynamics_Psi_u_050, label = "ψᵤ = 0.50")
plot!(months_range, Avg_income_dynamics_Psi_u_025, label = "ψᵤ = 0.25")
plot!(months_range, Avg_income_dynamics_Psi_u_005, label = "ψᵤ = 0.05")
title!("Income Dynamics")
xlabel!("Months Relative to Job Loss")
ylabel!("Monthly Income")
plot!(legend=:bottomright)
# savefig("Homework Two/Output/PS2_Image_06.png") 


Drop_050 = (1 - Avg_income_dynamics_Psi_u_050[24] / Avg_income_dynamics_Psi_u_050[6])*100
Drop_025 = (1 - Avg_income_dynamics_Psi_u_025[24] / Avg_income_dynamics_Psi_u_025[6])*100
Drop_005 = (1 - Avg_income_dynamics_Psi_u_005[24] / Avg_income_dynamics_Psi_u_005[6])*100