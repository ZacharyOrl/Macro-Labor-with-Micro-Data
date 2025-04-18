#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Homework Four

    Last Edit:  April 17, 2025
    Authors:    Zachary Orlando and Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, LaTeXStrings

#= ################################################################################################## 
    Part 2: Model Overview
    Consider a simplified form of the model from Huggett, Ventura, and Yaron [2011]. There are
    overlapping generations of households who live and for T periods (i..e, there is no retirement).
    Agents are heterogeneous in the human capital h, and assets k (there is a common level of ability).
    In each period, agents make a consumption savings decision and a decision about how much time
    to invest in their human capital, which follows a Ben-Porath structure.
=# ##################################################################################################

include("Tauchen_1986.jl")

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives

    T::Int64    = 30                            # Life-cycle to 30 years (annual)
    r::Float64  = 0.04                          # Interest rate  
    β::Float64  = 0.99                          # Discount rate  
    γ::Float64  = 2.0                           # Coefficient of Relative Risk Aversion 
    α::Float64  = 0.70                          # Human Capital Technology parameter 

    # Human Capital Grid
    h_min::Float64 = 1.0
    h_max::Float64 = 10.0
    nh::Int64      = 1000
    h_grid::Vector{Float64} = range(h_min, h_max, length=nh)   

    # Assets Grid
    k_min::Float64 = 0.0
    k_max::Float64 = 50.0
    nk::Int64      = 300
    k_grid::Vector{Float64} = range(k_min, k_max, length=nk)

    # Investing in Human Capital Grid
    s_min::Float64 = 0.0
    s_max::Float64 = 1.0
    ns::Int64      = 20
    s_grid::Vector{Float64} = range(s_min, s_max, length=ns)  

    R_grid::Vector{Float64} = [(1.0019)^(t - 1) for t in 1:30] # Rental rate on labor for an age t worker

    H              =  h_grid .+ (reshape(h_grid, :, 1) .* reshape(s_grid, 1, :)).^α    # H - Law of Human Capital pre-shock

    # Human Capital Shock
    nz::Int64      = 5
    μ::Float64     = -0.029                       # Mean of z process
    σ::Float64     = sqrt(0.11) #sqrt(0.21) #     # Standard deviation of z process

end 

# Initialize value function and policy functions
@with_kw mutable struct Results
    V::Array{Float64,3}
    k_policy::Array{Float64,3}
    k_policy_index::Array{Int64,3}
    s_policy::Array{Float64,3}
    s_policy_index::Array{Int64,3}
    c_policy::Array{Float64,3}
end

@with_kw struct OtherPrimitives
    Ω::Array{Int64, 3}
    Z_grid::Vector{Float64}
    Γ_z::Vector{Float64}
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    V               = zeros(T + 1, nh, nk)
    k_policy        = zeros(T, nh, nk)
    k_policy_index  = zeros(T, nh, nk)
    s_policy        = zeros(T, nh, nk)
    s_policy_index  = zeros(T, nh, nk)
    c_policy        = zeros(T, nh, nk)

    # Ω is an object that consider current Human Capital h and Studying/Investment s that maps into a value H(h,s).
    # However h' depends on the realizations of z in particular: h' = exp(z) * H(h,s).
    # We obtain the z nodes (z_grid) and weights (Γ_z) and then transform them into levels by Z_grid = exp.(z_grid)
    Ω           = zeros(nh, ns, nz)                 
    z_grid, Γ_z = tauchen(nz, 0.0, σ, μ) 
    Γ_z         = Γ_z[1,:]
    Z_grid      = exp.(z_grid)

    # Further, since me have a mapping from HxSxZ -> H, however this h' might not be in the h_grid, so we make an
    # adjustemnt using findnearest() we find the INDEX of h'

    for i in 1:nh
        for j in 1:ns
                Ω[i,j,:] = clamp.(searchsortedfirst.(Ref(h_grid), Z_grid .* H[i, j]), 1, nh)
        end
    end 
    
    other_param = OtherPrimitives(Ω, Z_grid, Γ_z)
    results     = Results(V, k_policy, k_policy_index, s_policy, s_policy_index, c_policy)

    return param, results, other_param
end

#= ################################################################################################## 
    2.1 Assignment

    Solve the model above and simulate a panel of indiviudals from the model using the suggested 
    parameters below and report the following.
       
    Functions

=# ##################################################################################################

function Flow_Utility(c::Float64, param::Primitives)
    @unpack_Primitives param                

    return (c^(1 - γ) - 1) / (1 - γ)
end 

function Solve_Problem(param::Primitives, results::Results, other_param::OtherPrimitives)
    # Solves the decision problem: param is the structure of parameters and results stores solutions 
    @unpack_Primitives param                
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    println("Begin solving the model backwards")
    for j in T:-1:1  # Backward induction
        println("Age is ", 24+j)

        #= --------------------------------- STATE VARIABLES ---------------------------------------- =#
        Threads.@threads for h_index in 1:nh                        # State: Human Capital h
            h = h_grid[h_index]

            for k_index in 1:nk                                     # State: Assets k
                k = k_grid[k_index]
                candidate_max = -Inf 
                Fin_Inc       = k * (1+ r)                    
        #= --------------------------------- DECISION VARIABLES ------------------------------------- =#
                for s_index in 1:ns                                 # Control: Investment s
                    s = s_grid[s_index]
                    # H =  h + (h * s)^α                            # H - Law of Human Capital pre-shock
                    Lab_Inc = R_grid[j] * h * (1-s)

                    for kp_index in 1:nk                            # Control: Assets k'
                        kp = k_grid[kp_index]
                        c = Fin_Inc + Lab_Inc - kp                  # Consumption
        #= --------------------------------- GRID SEARCH -------------------------------------------- =#
                        if c > 0                                    # Feasibility check
                            val = Flow_Utility(c, param)            # Flow utility

                            for zp_index in 1:nz                    # Recall Ω provides the index for h'
                                    val += β * Γ_z[zp_index] * V[j + 1, Ω[h_index, s_index, zp_index], kp_index] 
                            end

                            if val > candidate_max                  # Check for max
                                candidate_max                       = val
                                k_policy[j, h_index, k_index]       = kp
                                k_policy_index[j, h_index, k_index] = kp_index                              
                                s_policy[j, h_index, k_index]       = s
                                s_policy_index[j, h_index, k_index] = s_index                              
                                c_policy[j, h_index, k_index]       = c
                                V[j, h_index, k_index]              = candidate_max
                            end   
                        end
        #= ------------------------------------------------------------------------------------------- =# 

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

    @unpack_Primitives param                                             
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    # Initial distribution of human capital 

    # Random_h_draws     = rand(Normal(2.0, sqrt(0.5)), S)
    # Alternative initial dispersion of human capital for question d) 
    Random_h_draws     = rand(Normal(2.0, sqrt(2.5)), S) 

    h_indices          = max.(searchsortedlast.(Ref(h_grid), Random_h_draws),1)
    Initial_h_Dist     = h_grid[h_indices]         # Actual values from the grid

    # Distrbution of z shocks
    Γ_z_dist            = Categorical(Γ_z)
    z_shocks            = rand(Γ_z_dist, S, T) 

    # Outputs
    Human_Capital       = zeros(S, T) 
    Human_Capital_Index = zeros(Int64, S, T)
    Assets              = zeros(S, T) 
    Assets_Index        = zeros(Int64, S, T)
    Investing           = zeros(S, T) 
    Investing_Index     = zeros(Int64, S, T)
    Consumption         = zeros(S, T)
    Earnings            = zeros(S, T)

    for s = 1:S

        # Initial Human Capital
        Human_Capital_Index[s,1] = h_indices[s]
        Human_Capital[s,1]       = Initial_h_Dist[s]

        # Initial Savings      
        Assets[s,1]             = 0.0
        Assets_Index[s,1]       = 1

        # Initial Investing Policy 
        Investing[s,1]           = s_policy[1, Human_Capital_Index[s,1], Assets_Index[s,1]]
        Investing_Index[s,1]     = s_policy_index[1, Human_Capital_Index[s,1], Assets_Index[s,1]]

        # Consumpton
        Consumption[s, 1]        = c_policy[1, Human_Capital_Index[s,1], Assets_Index[s,1]]

        # Earnings
        Earnings[s, 1]           = R_grid[1] * Human_Capital[s, 1] * (1-Investing[s, 1])

        for t = 2:T 

            # Human capital evolution
            h_index     = Human_Capital_Index[s, t-1]
            s_index     = Investing_Index[s, t-1]
            k_index     = Assets_Index[s, t-1]
            zp_index    = z_shocks[s, t-1]

            Human_Capital_Index[s, t] = Ω[h_index, s_index, zp_index]
            Human_Capital[s, t]       = h_grid[Human_Capital_Index[s, t]]

            # Savings 
            Assets[s, t]              = k_policy[t, h_index, k_index]
            Assets_Index[s, t]        = k_policy_index[t, h_index, k_index]          

            # Investing 
            Investing[s, t]           = s_policy[t, h_index, k_index]
            Investing_Index[s, t]     = s_policy_index[t, h_index, k_index]          

            # Consumpton
            Consumption[s, t]         = c_policy[t, h_index, k_index]

            # Earnings
            Earnings[s, t]            = R_grid[t] * Human_Capital[s, t-1] * (1-Investing[s, t])
        end 

    end 

    return Human_Capital, Assets, Investing, Consumption, Earnings
end

#= ################################################################################################## 
    Plots
=# ##################################################################################################

#= ################################################################################################## 
    (a) Plot the average path of earnings in the model as well as the standard deviation, skewness 
    and kurtosis of earnings by age. How do these graphs compare data estimates you created in Part 
    (1) and those presented in Huggett, Ventura, and Yaron [2011].
=# ##################################################################################################


S = 10000
Human_Capital, Assets, Investing, Consumption, Earnings = simulate_model(param, results, other_param, S)

age           = 25:1:54

Mean_Earnings     = vec(mean(Earnings, dims=1))
Std_Dev_Earnings  = vec(std(Earnings, dims=1))
Skewness_Earnings = vec(mean(Earnings, dims=1) ./ median(Earnings, dims=1))
Kurtosis_Earnings = [kurtosis(Earnings[:, t]) for t in 1:T]

# Mean Earnings
plot(age, Mean_Earnings, label = "Mean Earnings")
title!("")
xlabel!("Age")
ylabel!("Mean Earnings")
plot!(legend=:topleft)
savefig("Homework Four/Output/PS4_Image_A01.png") 

# Standard Deviation Earnings
plot(age, Std_Dev_Earnings, label = "Standard Deviation Earnings")
title!("")
xlabel!("Age")
ylabel!("Standard Deviation Earnings")
plot!(legend=:topleft)
savefig("Homework Four/Output/PS4_Image_A02.png") 

# Skewness Earnings
plot(age, Skewness_Earnings, label = "Skewness Earnings")
title!("")
xlabel!("Age")
ylabel!("Skewness Earnings")
plot!(legend=:topleft)
savefig("Homework Four/Output/PS4_Image_A03.png") 

# Kurtosis Earnings
plot(age, Kurtosis_Earnings, label = "Kurtosis Earnings")
title!("")
xlabel!("Age")
ylabel!("Kurtosis Earnings")
plot!(legend=:topleft)
savefig("Homework Four/Output/PS4_Image_A04.png") 


#= ################################################################################################## 
    (b) Plot the policy function for investing in human capital as a function a function of (1) 
    assets and (2) human capital for workers of different ages.
=# ##################################################################################################

# Investing in Human Capital Policy Function: Assets
age       = [25, 35, 45, 51]
indices   = [ 1, 11, 21, 27]
HC_index = searchsortedlast(h_grid, quantile(vec(Human_Capital), 0.15))
plot(k_grid, s_policy[indices[1], HC_index, :], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(k_grid, s_policy[idx, HC_index, :], label = "t = $t")
end
title!("")
xlabel!("Assets")
ylabel!("Investing in Human Capital Policy Function")
plot!(legend=:bottomright)
savefig("Homework Four/Output/PS4_Image_B01.png") 

# Investing in Human Capital Policy Function: Human Capital
age       = [25, 35, 45, 51]
indices   = [ 1, 11, 21, 27]
K_index = searchsortedlast(k_grid, quantile(vec(Assets), 0.50))
plot(h_grid, s_policy[indices[1], :, K_index], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(h_grid, s_policy[idx, :, K_index], label = "t = $t")
end
title!("")
xlabel!("Human Capital")
ylabel!("Investing in Human Capital Policy Function")
plot!(legend=:bottomleft)
savefig("Homework Four/Output/PS4_Image_B02.png") 

#= ################################################################################################## 
    (c) How do these policy functions change if you increase the variance of shocks to human capital? 
    Why do you think you see this pattern?
=# ##################################################################################################

# Investing in Human Capital Policy Function: Assets
age       = [25, 35, 45, 51]
indices   = [ 1, 11, 21, 27]
HC_index = searchsortedlast(h_grid, 3.66)
plot(k_grid, s_policy[indices[1], HC_index, :], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(k_grid, s_policy[idx, HC_index, :], label = "t = $t")
end
title!("")
xlabel!("Assets")
ylabel!("Investing in Human Capital Policy Function")
plot!(legend=:bottomright)
savefig("Homework Four/Output/PS4_Image_C01.png") 

# Investing in Human Capital Policy Function: Human Capital
age       = [25, 35, 45, 51]
indices   = [ 1, 11, 21, 27]
K_index = searchsortedlast(k_grid, 6.36)
plot(h_grid, s_policy[indices[1], :, K_index], label = "t = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(h_grid, s_policy[idx, :, K_index], label = "t = $t")
end
title!("")
xlabel!("Human Capital")
ylabel!("Investing in Human Capital Policy Function")
plot!(legend=:bottomleft)
savefig("Homework Four/Output/PS4_Image_C02.png") 


#= ################################################################################################## 
    (d) Create a measure of lifetime earnings based upon Guvenen et al. [2017]. If you increase the 
    initial dispersion of human capital, how does your measure of lifetime inequality change? How 
    does the path of the standard deviation of earnings by age compare to your graph from part (a)?
=# ##################################################################################################

S = 10000
Human_Capital, Assets, Investing, Consumption, Earnings = simulate_model(param, results, other_param, S)
Lifetime_Earnings_1 = vec(mean(Earnings, dims=2))
Std_Dev_Earnings_1  = vec(std(Earnings, dims=1))

S = 10000
Human_Capital, Assets, Investing, Consumption, Earnings = simulate_model(param, results, other_param, S)
Lifetime_Earnings_2 = vec(mean(Earnings, dims=2))
Std_Dev_Earnings_2  = vec(std(Earnings, dims=1))


# Histogram of lifetime earnings
histogram([Lifetime_Earnings_1, Lifetime_Earnings_2],
          label  = [L"σ^2 = 0.50" L"σ^2 = 2.50"],
        #   title  = "Histogram of Lifetime Earnings Observations",
          xlabel = "Lifetime Earnings",
          bins   = 50, 
          legend =:topleft,
          alpha  = 0.6)
savefig("Homework Four/Output/PS4_Image_D01.png") 

# Standard Deviation Earnings
plot(age, Std_Dev_Earnings_1, label = L"Standard \; Deviation \; Earnings \; with \; σ^2 = 0.50")
plot!(age, Std_Dev_Earnings_2, label = L"Standard \; Deviation \; Earnings \; with \; σ^2 = 2.50")
title!("")
xlabel!("Age")
ylabel!("Standard Deviation Earnings")
plot!(legend=:bottomright)
savefig("Homework Four/Output/PS4_Image_D02.png") 