#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Homework Two

    Last Edit:  April 3, 2025
    Authors:    Zachary Orlando and Cutberto Frias Sarraf

=# ##################################################################################################

#= ################################################################################################## 
    Part 1: Data
    In this section, we are going to produce a moment from the PSID earnings data that we will 
    compare to the estimates from the model in Section 2. Using the PSID data, individuals who have 
    been working full-time in two consecutive years (easy way to do this, set a lower bound on annual 
    hours that aligns with working full-time for a full year). What is the average change in earnings 
    for these individuals?
=# ##################################################################################################

# Need to place the path to the below stata file on your computer. 
stata_path = "C:/Program Files/Stata18/StataSE-64.exe"

# Add a path to the dataset "pequiv_long.dta" AND the dataset on annual price levels. 
data_dir = "/Users/cutbertofs/Documents/UW/VSC/810 ECON Carter/Macro-Labor-with-Micro-Data/Homework Two/" 

# Set folders where the results and images will be outputted 
output = "$data_dir/output"

#= ################################################################################################## 
    I. PS2_Clean_PSID.do 
    Obtain the average change in earnings for individuals who have been working full-time in two 
    consecutive years using PSID data.
=# ##################################################################################################

cd(data_dir)
println("Running Stata script: PS2_Clean_PSID.do")
run(`$stata_path -b do $data_dir/PS2_Clean_PSID.do`)

#= ################################################################################################## 
    II. Part_1_Data.jl
    Simulate data on 500 job losers and 500 job stayers for 11 years. We estimate a distributed lag 
    specification.
=# ##################################################################################################

println("Running Julia script: Part_1_Data.jl")
include("Part_1_Data.jl")

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

#= ################################################################################################## 
    III. Part_2_Model.jl
    Solve the model with VFI and simulate a mass of agents.
=# ##################################################################################################

println("Running Julia script: Part_2_Model.jl")
include("Part_2_Model.jl")