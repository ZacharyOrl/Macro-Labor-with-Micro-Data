#= ################################################################################################## 
    Econ 810: Spring 2025 Advanced Macroeconomics 
    Authors:    Zachary Orlando and Cutberto Frias Sarraf
=# ##################################################################################################

using Plots, Random, LinearAlgebra, Statistics, DataFrames, GLM, CategoricalArrays, FixedEffectModels, CSV

#= ################################################################################################## 
    Part 1: Data
    1.2 Earnings losses while unemployed
    In this section, you will simulate data for a sample of “job losers” and job stayers. Then using 
    the simulated data, you will test that you can recover the parameters that govern the simulation 
    using the distributed lag framework from class. Try the following:

    • Step 1: Simulate data on 500 job losers and 500 job stayers for 11 years. In year 1, suppose 
    all individuals make $30,000 plus some random (mean zero) noise. In years 2-5 (the pre- layoff 
    years), suppose all individuals earnings increase by $1000 + random noise. For the job losers 
    sample suppose in year 6 their earnings decline by 9,000 plus noise, and then resume increasing 
    by 1000 per year (plus noise) until year 11. For the job stayers sample suppose in year 6 
    through 11 they continue to have an average increase in earnings of 1000 (plus mean zero noise).
=# ##################################################################################################

function Simulate_Earnings(N, T, X)
    
    # Initialize matrices
    JL = zeros(N, T)  # Job Losers
    JS = zeros(N, T)  # Job Stayers

    # Standard deviation for the initial earnings shock
    σ₀ = 1000

    # Standard deviation for earnings growth
    σ₁ = 500

    # Assigning the workers first year's earnings 
    JL[:, 1] = 30000 .+ σ₀ * randn(N)
    JS[:, 1] = 30000 .+ σ₀ * randn(N)

    # Simulate the earnings trayectories for these workers
    for i in 2:T

        if i != X
            JL[:, i] = JL[:, i-1] .+ 1000 .+ σ₁ * randn(N)  # 
            JS[:, i] = JS[:, i-1] .+ 1000 .+ σ₁ * randn(N)  # 
        else
            JL[:, i] = JL[:, i-1] .- 9000 .+ σ₁ * randn(N)  # 
            JS[:, i] = JS[:, i-1] .+ 1000 .+ σ₁ * randn(N)  # 
        end

    end

    return JL, JS
end

# Parameter values
N = 500
T = 11
X = 6

JL, JS = Simulate_Earnings(N, T, X)

mean_JL = mean(JL, dims = 1)[1,:]
mean_JS = mean(JS, dims = 1)[1,:] 

#= ################################################################################################## 
    Plots
=# ##################################################################################################

plot(1:11, mean_JL, label = "Average Earnings for Job Losers")
plot!(1:11, mean_JS, label = "Average Earnings for Job Stayers")
title!("Average Earnings")
xlabel!("Years")
ylabel!("Average Earnings")
plot!(legend=:topleft)

#= ################################################################################################## 
    • Step 2: Using the simulated data from step 1, estimate the distributed lag framework from class,
    where αi is an individual fixed effect , γt are year fixed effects, Di,k are dummy variables 
    denoted when an individual i is k years from layoff.

    To complete the data assignment report the coefficient estimates from estimating the distributed 
    lag framework. What are the values of βk and γt ? What do you think they should be equal to?
=# ##################################################################################################

# Create vectors for individual ID and year
individuals_JL = repeat(1:N, inner=T)          # Repeats each ID for all years
individuals_JS = repeat(N+1:2N, inner=T)       # Repeats each ID for all years
years          = repeat(1:T, outer=N)          # Assigns year 1:T to each individual

# Convert JL and JS matrices to long format
earnings_JL = reshape(JL', :)  # Transpose and then vectorize
earnings_JS = reshape(JS', :)  # Transpose and then vectorize

# Create group labels
group_JL = fill("JL", N * T)
group_JS = fill("JS", N * T)

# Create DataFrames
Data_JL       = DataFrame(Individual=individuals_JL, Year=years, Earnings=earnings_JL, Group=group_JL)
Data_JS       = DataFrame(Individual=individuals_JS, Year=years, Earnings=earnings_JS, Group=group_JS)
Data_Earnings = vcat(Data_JL, Data_JS)

# Convert categorical variables
Data_Earnings.Individual  = CategoricalArray(Data_Earnings.Individual)
Data_Earnings.Year        = CategoricalArray(Data_Earnings.Year)
Data_Earnings.Group       = CategoricalArray(Data_Earnings.Group)
first(Data_Earnings, 10)

# Estimate the regression
model           = reg(Data_Earnings, @formula(Earnings ~ 0 + fe(Individual) + Year + Group*Year))
print(model)

#= ################################################################################################## 
    Additional Plots
=# ##################################################################################################

D_ik = [-9.185664, -43.48981, 24.57907, 17.692, 0.0, -10025.85, -10002.07, 
                        -9941.054, -9894.838, -9902.49, -9907.042]

plot(-5:1:5, D_ik,
    label = "Earnings Loss",
    title = "Earnings Loss",
    xlabel = "Years (Year 6 is the layoff year)",
    ylabel = "Earnings Loss",
    legend = :topright,
    lw = 2,
    marker = :circle)
# savefig("Homework Two/Output/PS2_Figure_01.png") 


##################################################################################################

# Data_Earnings.Interaction = CategoricalArray(string.(Data_Earnings.Group, "_", Data_Earnings.Year))
# Data_Earnings.Interaction = CategoricalArray(
#     [
#         Data_Earnings.Group[i] == "JL" ? "JL" : string(Data_Earnings.Group[i], "_", Data_Earnings.Year[i])
#         for i in 1:nrow(Data_Earnings)
#     ]
# )

# # Specify a valid base level from the interaction variable (e.g., "JL_1")
# base_level = "JS_4"  # Replace with the actual base level you want to use

# # Fit the model with Effects Coding using the specified base level
# model = reg(Data_Earnings, @formula(Earnings ~ 0 + Individual + Year + Interaction), 
#             contrasts = Dict(:Interaction => EffectsCoding(base=base_level)))


# Data_Earnings.Interaction = CategoricalArray(string.(Data_Earnings.Group, "_", Data_Earnings.Year))
# Data_Earnings.JL         = ifelse.(Data_Earnings.Group .== "JL", 1, 0)  # Dummy for Job Losers (JL)
# Data_Earnings.JS         = ifelse.(Data_Earnings.Group .== "JS", 1, 0)  # Dummy for Job Stayers (JS)
# for y in levels(Data_Earnings.Year)
#     Data_Earnings[!, Symbol("D_" * string(y))] = (Data_Earnings.Year .== y) .& (Data_Earnings.JL .== 1)
# end

# base_level = "D_5"  # Replace with the actual base level you want to use

# Estimate the regression
# model = reg(Data_Earnings, @formula(Earnings ~ 0 + Year + Individual + D_1 + D_2 + D_3 + D_5 + D_6 + D_7 + D_8 + D_9 + D_10 + D_11))
# CSV.write("Homework Two/Output/Dataframe_Simulations.csv", Data_Earnings)

# coefficients    = coef(model)
# coeffnames      = coefnames(model)







