###########################################
# Estimates the Bewley model with Housing
# using income as a sum of a deterministic process, 
# a persistent (AR1) income process and a transitory process.  
# housing follows a deterministic trend. 
###########################################
# Packages 
###########################################
using Parameters, Interpolations, CSV, Plots,Distributions,LaTeXStrings,Statistics, DataFrames, LinearAlgebra, Optim, Roots
###########################################
# Put your path to the dataset "estimation_results.csv" below 
#indir = "C:/Users/zacha/Documents/2025 Spring/Advanced Macroeconomics/J. Carter Braxton/Homework/ZO_Project/"

outdir_images = "/home/z/zorlando/ZO_Project/images"
indir_parameters = "/home/z/zorlando/ZO_Project/parameters"

###########################################
# Parameters
###########################################
cd(indir_parameters)

@with_kw struct Model_Parameters @deftype Float64

    # Housing parameters
    b::Float64 = 0.01 # Annual Log House Price increase used in Cocco (2005) - he estimates 0.016 but uses a lower number to account for quality improvement. 
    d::Float64 = 0.2                               # Limit on Home equity you can take out in any period. 
    #π::Float64 = compound(1 + 0.032, 5) - 1        # Moving shock probability 
    #δ::Float64 = compound(1 + 0.01, 5) - 1         # Housing Depreciation
    λ::Float64 = 0.05                               # House-transaction cost
    γ::Float64 = 0.2                                # Down-payment proportion 


    # Rent cost: 
    # In 2000 it was $602 a month and home values were $119,600 so about renters spend about 6% of the typical home value renting 
    # In 1960, the typical house price was 58,600 (in 2000 dollars)
    # Source: https://www2.census.gov/programs-surveys/decennial/tables/time-series/coh-values/values-adj.txt
    rent_prop::Float64 = 0.05
    P_bar::Float64 = 65300 #About 2 x median household income in 1960 https://kinder.rice.edu/urbanedge/can-texas-afford-lose-its-housing-affordability-advantage

    # Utility function parameters
    # θ::Float64 = 0.1              # Utility from housing services relative to consumption
    σ = 2.0 # Coefficient of Relative Risk Aversion 
    β = 0.975 # Discount rate 
    
    R_F = 0.04 # Assumed risk-free interest rate 
    R_M = 0.07 # Mortgage interest rate
    
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
    ζ_grid::Vector{Float64} = rouwenhorst(σ_ζ/(1 - φ^2), φ, 10)[1]
    T_ζ::Matrix{Float64} = rouwenhorst(σ_ζ/(1 - φ^2), φ, 10)[2]

    nζ::Int64 = length(ζ_grid)

    # Initial Persistent Component 
    σ_0_grid::Array{Float64,1}  = compute_initial_dist(0.0, sqrt(0.15), ζ_grid)

    # Transitory Component 
    ϵ_grid::Vector{Float64} = rouwenhorst(σ_ϵ, 0.0, 5)[1]
    T_ϵ::Matrix{Float64} = rouwenhorst(σ_ϵ, 0.0, 5)[2]

    nϵ::Int64 = length(ϵ_grid)

    # Load the lifecycle component of income
    κ::Matrix{Float64} = vcat(hcat(CSV.File("life_cycle_income.csv").age,CSV.File("life_cycle_income.csv").deterministic_component),[60.0 0.0])
    
    # Punishment for defaulting 
    pun::Float64 = -10^8 

    # Cash-on-Hand grids
    na::Int64 = 100 # Number of cash on hand grid points 

    # Mortgage Debt
    # Not interested in studying the impact of house price risk on consumption 
    # So House Prices are deterministic. 
    M_min = 0.0 
    M_max = (1-γ) * P_bar * exp(b * (N-1))
    nM::Int64 = 10 # Number of mortgage debt grid points 

    M_grid::Array{Float64,1} = collect(range(M_min, length = nM, stop = M_max))

    # Housing levels: 0 = rent, 1 = own
    H_grid::Vector{Int64} = [0, 1]
    nH::Int64 = 2
end 

#initialize value function and policy functions
mutable struct Solutions

    val_func::Array{Float64,6}
    H_pol_func::Array{Float64,6}
    a_pol_func::Array{Float64,6}
    M_pol_func::Array{Float64,6}
    D_pol_func::Array{Float64,6}
    # Housing taste parameters
    # For now 0, when housing taste is introduced it will be 0.2 
    # Following Paz-Pardo - I set the housing utility share ν to 0.2, based on NIPA data on budget shares
    θ::Float64 

    # Proportional utility increase from owning rather than renting (1 = no increase)
    s::Float64

    # a Grids 
    a_grids::Array{Float64,2}

end

function build_solutions(para) 

    val_func    = zeros(Float64, para.nH, para.nϵ, para.nζ, para.nM, para.na, para.N + 1 ) 
    H_pol_func  = zeros(Float64, para.nH, para.nϵ, para.nζ, para.nM, para.na, para.N ) 
    M_pol_func  = zeros(Float64, para.nH, para.nϵ, para.nζ, para.nM, para.na, para.N ) 
    a_pol_func  = zeros(Float64, para.nH, para.nϵ, para.nζ, para.nM, para.na, para.N ) 
    D_pol_func  = zeros(Float64, para.nH, para.nϵ, para.nζ, para.nM, para.na, para.N ) 
    θ = 0.0 
    s = 1.0

    # Make a different grid for each period, 
    # to reflect that house prices rise each period.
    a_grids = zeros(para.na,para.N + 1)

    for j = 1:para.N + 1
        a_min = 0.01
        a_max = 1000000  # Maximum assets on the grid rises
        a_grids[:,j] = collect(range(start = a_min, length = para.na, stop = a_max)) 
    end 

    sols = Solutions(val_func,H_pol_func,a_pol_func, M_pol_func, D_pol_func, θ, s, a_grids)

    return sols
end 

function Initialize_Model() 

    para = Model_Parameters()
    sols = build_solutions(para)

    return para, sols 
end
###########################################
# Auxilary Functions
###########################################
# cd(indir)
# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

# Function which computes an initial distribution over a given grid 
include("compute_initial_dist.jl")

# Functions which solves the lifecycle problem of the agent
include("Solve_Problem_a.jl")

# Function which simulates the model
include("simulate_a_model.jl")

# Function which calibrates the model 
include("Calibrate_Model.jl")

# Function which omputes insurance coefficients
include("compute_insurance.jl")

# Function which omputes insurance coefficients
include("Solve_Model_with_Debt.jl")


#########################################################
# Functions 
#########################################################
function flow_utility_func(c::Float64, H_index::Int64, para::Model_Parameters, sols::Solutions)
    @unpack σ = para
    @unpack s, θ = sols

    # Homeowners gain additional utility from housing services
    if H_index == 2
        return (    ( c^(1-θ) * s^θ )^( 1 - σ )   ) / (1 - σ)

    else 
        
        return (    ( c^(1-θ) )^( 1 - σ )   ) / (1 - σ)
    end 
end 

# Takes as input all states and choices necessary to pin down the budget constraint
# and outputs the sum of bonds (cannot short-sell bonds)
# If there is a house trade (buying or selling) then you must pay an adjustment cost.
# I assume mortgages are portable.
function budget_constraint( ap::Float64, coh::Float64, M_index::Int64, M_prime_index::Int64,
                            H_index::Int64, H_prime_index::Int64, P::Float64, para::Model_Parameters)
    @unpack R_M, λ, rent_prop, M_grid = para

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]

    # If there is no house trade and you own a home
    if H_prime_index == 2 && H_index == 2
       c = coh - ap - R_M * M - (M - M_prime)  # M_prime < M /(1 + r)
    end 
    # If there is no house trade and you rent
    if H_prime_index == 1 && H_index == 1
        c = coh - ap - rent_prop * P # M_prime < M /(1 + r)
    end 
    
    # Buying a home
    if H_prime_index == 2 && H_index == 1
        c = coh - ap - (1+λ) * (P) + M_prime 
    end 

    # Selling a home and renting
    if H_prime_index == 1 && H_index == 2
        c = coh - ap - M + (1 - λ) * P  - rent_prop * P 
    end

    return c
end 

# Reports the difference between debt taken out today and the collateral limit: 
function mortgage_constraint(M_index::Int64, M_prime_index::Int64, H_index::Int64, H_prime_index::Int64,P::Float64, t::Int64, para::Model_Parameters)
    @unpack_Model_Parameters para
    
    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]

    # Holding a home
    if H_index == 2 && H_prime_index == 2
        out = M - M_prime # Can't accrue any more debt and must pay interest. 
    end 

    # Buying a home
    if H_prime_index == 2 && H_index == 1
       out =  (1 - γ) * P  - M_prime  # Mortgage cannot exceed (1-γ)% of the home value
    end 

    # Selling a home
    if H_prime_index == 1 && H_index == 2
       out =  - M_prime  # Must pay off mortgage when selling home
    end 

    # Renting
    if H_prime_index == 1 && H_index == 1
       out =  - M_prime  # Renters will not hold a mortgage
    end 

    # In the terminal period, you must pay back your mortgage in its entirety. 
    if t == N 
        out = -1 * M_prime 
    end 

    return out
end 

# For HELOC evaluation
# Reports the difference between debt taken out today and the collateral limit: 
function debt_constraint(D::Float64, H_prime::Float64, P::Float64,para::Model_Parameters)
    @unpack_Model_Parameters para

    return (1-d) * H_prime * P - D
end 

# Creates a linear interpolation function using a mapping from grid x1 to outcome F. 
#
# Allows for extrapolation outside the grid (Flat extrapolation)
function linear_interp(F::Vector{Float64}, x1::Vector{Float64})
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))

    interp = interpolate(F, BSpline(Linear()))
    scaled_interp = Interpolations.scale(interp,x1_grid)
    extrap = extrapolate(scaled_interp, Interpolations.Flat())
    return  extrap
end

###################################################
# Solve the model and simulate 
###################################################
para,sols = Initialize_Model()
Solve_Problem(para,sols)
wealth, assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing = simulate_model(para, sols, 10000)

###################################################
# Plots no housing taste 
###################################################
cd(outdir_images)
start_age = 25 
end_age = 60

age_grid = collect(range(start_age, length = end_age - start_age + 1, stop = end_age))

# Plot wealth statistics by age 
plot(age_grid, mean(wealth,dims = 1)[1,:], label = "Mean")
plot!(age_grid, median(wealth,dims = 1)[1,:],label = "Median",xlabel = "Age", ylabel = L"Wealth\n (1997 $) ", title = "Wealth Accumulation over the Lifecycle")
plot!(age_grid, mean(assets,dims = 1)[1,:],label = "Mean Savings",xlabel = "Age",title = "Wealth Accumulation over the Lifecycle")
savefig("ZO_Project_Image_01.png") 

# Plot consumption variance by age 
plot(age_grid, (var(log.(consumption),dims = 1)[1,:]),xlabel = "Age", ylabel = "Variance of logs",label = "",title = "Consumption Inequality by Age")
savefig("ZO_Project_Image_02.png") 

# Plot the histogram of assets for the age with the highest average wealth - to ensure the upper bound is reasonable
# for this starting value. 
histogram(wealth[:,20],xlabel = L"Wealth ($)",label ="Wealth", title = "Distribution of Wealth at Age 45",normalize = :probability)
histogram!(assets[:,20],xlabel = L"Wealth ($)",label ="Savings", title = "Distribution of Wealth at Age 45",normalize = :probability)
savefig("ZO_Project_Image_03.png") 

# Plot the wealth of homeowners versus non-homeowners at 45
wealth_homeowners = ifelse.(housing .== 1, wealth, missing)
wealth_renter = ifelse.(housing .== 0, wealth, missing)

plot(age_grid, mapslices(x -> mean(skipmissing(x)), wealth_homeowners; dims=1)[1,:], label = "Homeowner")
plot!(age_grid, mapslices(x -> mean(skipmissing(x)), wealth_renter; dims=1)[1,:],xlabel = "Age", label = "Renter",ylabel = L" Mean Wealth\n (1997 $) ", title = "Wealth Accumulation over the Lifecycle")
savefig("ZO_Project_Image_04.png") 


# Plot the proportion of individuals who are homeowners in each period 
plot(age_grid, mean(housing,dims = 1)[1,:], label = "", xlabel = "Age", ylabel = "Proportion Homeowners", title = "Homeownership Rate by Age")
savefig("ZO_Project_Image_05.png") 

# Finally, plot consumption inequality within groups 
consumption_homeowners = ifelse.(housing .== 1, consumption, missing)
consumption_renters = ifelse.(housing .== 0, consumption, missing)

plot(age_grid, mapslices(x -> var(log.(skipmissing(x))), consumption_homeowners; dims=1)[1,:],xlabel = "Age", ylabel = "Variance of logs",label = "Homeowners",title = "Consumption Inequality by Age")
plot!(age_grid, mapslices(x -> var(log.(skipmissing(x))), consumption_renters; dims=1)[1,:],xlabel = "Age", ylabel = "Variance of logs",label = "Renters",title = "Consumption Inequality by Age")
savefig("ZO_Project_Image_06.png") 

######################################
# Add Housing Taste and Calibrate Model 
######################################
sols.θ  = 0.2  # Using the same housing taste value used in Paz-Pardo 
data_moment = 0.791 # Homeownership rate of households with head aged 45 between 1970 and 1998 in the SCF. 

calibrate_model(data_moment,para,sols) # s = approx 1.03 hits the moment 
wealth, assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing = simulate_model(para, sols, 10000)

###################################################
# Plots with Housing Taste
###################################################
cd(outdir_images)
start_age = 25 
end_age = 60

age_grid = collect(range(start_age, length = end_age - start_age + 1, stop = end_age))

# Plot wealth statistics by age 
plot(age_grid, mean(wealth,dims = 1)[1,:], label = "Mean")
plot!(age_grid, median(wealth,dims = 1)[1,:],label = "Median",xlabel = "Age", ylabel = L"Wealth\n (1997 $) ", title = "Wealth Accumulation over the Lifecycle")
plot!(age_grid, mean(assets,dims = 1)[1,:],label = "Mean Savings",xlabel = "Age",title = "Wealth Accumulation over the Lifecycle")
savefig("ZO_Project_Image_01.png") 

# Plot consumption variance by age 
plot(age_grid, (var(log.(consumption),dims = 1)[1,:]),xlabel = "Age", ylabel = "Variance of logs",label = "",title = "Consumption Inequality by Age")
savefig("ZO_Project_Image_02.png") 

# Plot the histogram of assets for the age with the highest average wealth - to ensure the upper bound is reasonable
# for this starting value. 
histogram(wealth[:,20],xlabel = L"Wealth ($)",label ="Wealth", title = "Distribution of Wealth at Age 45",normalize = :probability)
histogram!(assets[:,20],xlabel = L"Wealth ($)",label ="Savings", title = "Distribution of Wealth at Age 45",normalize = :probability)
savefig("ZO_Project_Image_03.png") 

# Plot the wealth of homeowners versus non-homeowners at 45
wealth_homeowners = ifelse.(housing .== 1, wealth, missing)
wealth_renter = ifelse.(housing .== 0, wealth, missing)

plot(age_grid, mapslices(x -> mean(skipmissing(x)), wealth_homeowners; dims=1)[1,:], label = "Homeowner")
plot!(age_grid, mapslices(x -> mean(skipmissing(x)), wealth_renter; dims=1)[1,:],xlabel = "Age", label = "Renter",ylabel = L" Mean Wealth\n (1997 $) ", title = "Wealth Accumulation over the Lifecycle")
savefig("ZO_Project_Image_04.png") 


# Plot the proportion of individuals who are homeowners in each period 
plot(age_grid, mean(housing,dims = 1)[1,:], label = "", xlabel = "Age", ylabel = "Proportion Homeowners", title = "Homeownership Rate by Age")
savefig("ZO_Project_Image_05.png") 

# Finally, plot consumption inequality within groups 
consumption_homeowners = ifelse.(housing .== 1, consumption, missing)
consumption_renters = ifelse.(housing .== 0, consumption, missing)

plot(age_grid, mapslices(x -> var(log.(skipmissing(x))), consumption_homeowners; dims=1)[1,:],xlabel = "Age", ylabel = "Variance of logs",label = "Homeowners",title = "Consumption Inequality by Age")
plot!(age_grid, mapslices(x -> var(log.(skipmissing(x))), consumption_renters; dims=1)[1,:],xlabel = "Age", ylabel = "Variance of logs",label = "Renters",title = "Consumption Inequality by Age")
savefig("ZO_Project_Image_06.png") 

######################################
# Compute insurance coefficients and plot them over time
######################################
α_ϵ = zeros(7,1)
α_ζ = zeros(7,1)
for j = 1:5:para.N
    index = Int64((j+4)/5)
    α_ϵ[index],α_ζ[index] = true_insurance(sols,transitory[:,j:j+4],persistent[:,j:j+4],consumption[:,j:j+4], para.φ) 
end

age_grid = collect(range(start_age, length = 7, stop = end_age))
plot(age_grid,α_ϵ,xlabel = "Age", ylabel = "Value", label = "Insurance against Transitory Shocks")
plot!(age_grid,α_ζ, label =  "Insurance against Persistent Shocks")

# Need to compare to insurance under the no-housing model. 
# and for homeowners vs renters. 

α_ϵ,α_ζ = true_insurance(sols,transitory[:,1:35],persistent[:,1:35],consumption[:,1:35], para.φ) 
############################################################
# Policy evaluation: HELOC
############################################################
Solve_Problem_with_Heloc(para,sols)

wealth, assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing = simulate_model(para, sols, 10000)
α_ϵ,α_ζ = true_insurance(sols,transitory[:,1:35],persistent[:,1:35],consumption[:,1:35], para.φ) 
# HELOC doesn't improve insurance all that much: individuals can only access it if 
# they are least in need. 
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

