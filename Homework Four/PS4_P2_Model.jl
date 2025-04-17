#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Homework Four

    Last Edit:  April 17, 2025
    Authors:    Zachary Orlando and Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, DataFrames,FastGaussQuadrature
using CategoricalArrays, StatsPlots

#= ################################################################################################## 
    Part 2: Model Overview
    Consider a simplied form of the model from Huggett, Ventura, and Yaron [2011]. There are
    overlapping generations of households who live and for T periods (i..e, there is no retirement).
    Agents are heterogeneous in the human capital h, and assets k (there is a common level of ability).
    In each period, agents make a consumption savings decision and a decision about how much time
    to invest in their human cpaital, which follows a Ben-Porath structure.
=# ##################################################################################################

include("Homework Two/Tauchen_1986.jl")

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives

    T::Int64    = 30                            # Life-cycle to 30 years (annual)
    r::Float64  = 0.04                          # Interest rate  
    β::Float64  = 0.99                          # Discount rate  
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


