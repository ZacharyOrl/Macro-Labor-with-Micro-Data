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
    r        = 0.04                          # Interest rate  
    β        = 0.99                          # Discount rate  
    σ        = 2.0                           # Coefficient of Relative Risk Aversion 
    δ        = 0.1                           # Job-Destruction Rate

    τ        = 0.2                           # Marginal tax rate on wages
    ζ        = 1.6                           # Matching elasticity parameter
    κ        = .995                          # Cost of posting a vacancy for a firm
    z        = 0.4                           # Unemployment Benefit (Transfer funded by τ)

    p_hl     = 0.5                           # Per-period probability of moving down one human capital grid point when unemployed
    p_hl     = 0.05                           # Per-period probability of moving up one human capital grid point when employed

    # Grids
    # Human Capital
    h_min = 0.5
    h_max = 1.5
    nh::Int64      = 25
    h_grid::Vector{Float64} = range(h_min, h_max, length=nh)   

    # Piece Wages 
    w_min = 0.0
    w_max = 1.0
    nw::Int64      = 25
    w_grid::Vector{Float64} = range(w_min, w_max, length=wh)   

    # Saving  - use b to match assignment notation
    b_min = 0.0  # (Default Calibration assumes a ZBC)
    b_max = 100.0
    nb::Int64      = 41
    b_grid::Vector{Float64} = range(b_min, b_max, length=nb)

end 

#initialize value function and policy functions
@with_kw mutable struct Results
    U::Array{Float64,3}             # U[T, b, h]
    W::Array{Float64,4}             # V[T, w, b, h]
    J::Array{Float64,3}             # V[T, w, h]

    W_policy::Array{Float64,2}      # Once matched, the only choice is b so pol[T, b]
    U_policy::Array{Float64,3}      # Need to choose a w to search in and a b so pol[T, w, b]

    θ::Array{Float64,3}             # Market Tightness of each submarket for each period θ[T, w, h]
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    U          = zeros(T + 1, nb, nh)
    W          = zeros(T + 1, nw, nb, nh) 
    J          = zeros(T + 1, nw, nh) 

    W_policy   = zeros(T, nb)
    U_policy   = zeros(T, nw, nb)       
    θ          = zeros(T, nw, nh)

    results  = Results(U, W, J, W_policy, U_policy, θ)
    return param, results
end