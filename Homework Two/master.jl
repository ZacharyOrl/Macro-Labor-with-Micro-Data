#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Homework Two

    Last Edit:  April 3, 2025
    Authors:    Zachary Orlando and Cutberto Frias Sarraf

=# ##################################################################################################

#= ################################################################################################## 
    Part 1: Data
    1.1 Earnings gains while employed
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
    1. PS2_Clean_PSID.do
    
    Obtain the average change in earnings for individuals who have been working full-time in two 
    consecutive years using PSID data.
=# ##################################################################################################

cd(data_dir)
println("Running Stata script: PS2_Clean_PSID.do")
run(`$stata_path -b do $data_dir/PS2_Clean_PSID.do`)

#= ################################################################################################## 
    2. Part_1_Data.jl
    
    Simulate (...)
=# ##################################################################################################

println("Running Julia script: Part_1_Data.jl")
include("Part_1_Data.jl")

#= ################################################################################################## 
    3. X.jl
    
    X
=# ##################################################################################################

