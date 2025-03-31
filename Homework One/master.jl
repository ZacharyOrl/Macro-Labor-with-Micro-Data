################################################################
# Econ 810: Spring 2025 Advanced Macroeconomics 
# This file implements the scripts needed to replicate the
# calibration of the Bewley model using PSID data 
# in Homework One. 

# Authors: Zachary Orlando and Cutberto Frias Sarraf
################################################################
# Inputs 

# Need to place the path to the below stata file on your computer. 
stata_path = "C:/Program Files/Stata18/StataSE-64.exe"

# Put your path to the datasets "pequiv_long.dta" AND a dataset on annual price levels below 
# I use annual CPI for All Urban Consumers in the US indexed to 2024. (called "CPI.csv") 
data_dir = "C:/Users/zacha/Documents/2025 Spring/Advanced Macroeconomics/J. Carter Braxton/Homework/Homework One" 

# Set folders where the results and images will be outputted 
outdir_parameters = "$data_dir/parameters"
outdir_images = "$data_dir/images"
############################################################
#= 1. clean_psid.do
   
 Creates a HH-year panel of predictable and residual income from the PSID.
 Generates sample summary tables and lifecycle predictable income plot. 
=#
cd(data_dir)
println("Running Stata script: clean_psid.do")
run(`$stata_path -b do $data_dir/clean_psid.do`)

#= 2. est_var_params.jl
 
 Takes the residual income from the panel and estimates the variance of 
 transitory & persistent income using Generalized Method of Moments (GMM).   
=#
println("Running Julia script: est_var_params.jl")
include("est_var_params.jl")

#= 3. discretize_income_processes.jl
 
 Using the estimated variance parameters from 2., discretizes 
 permanent and transitory income processes into a grid 
 and transition probabilities using the Rouwenhorst method. 

 Generates plots of their stationary distributions as checks. 
=#
println("Running Julia script: discretize_income_processes.jl")
include("discretize_income_process.jl")

#= 4. bewley_model.jl
 
 Using the discretized processes from 3. and estimated predictable income  
 from 2., solves a Bewley model with persistent and transitory income 
 risk and then simulates the model.  

 Generates plots of the simulated distribution and insurance parameters
 using the BPP (2008) method and the true values as per Kaplan & Violante (2010). 
=#
println("Running Julia script: bewley_model.jl")
include("bewley_model.jl")

println("All scripts executed successfully.")