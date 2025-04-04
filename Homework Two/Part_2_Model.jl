#= ################################################################################################## 
    Econ 810: Spring 2025 Advanced Macroeconomics 
    Authors:    Zachary Orlando and Cutberto Frias Sarraf
=# ##################################################################################################

using Plots, Random, LinearAlgebra, Statistics, DataFrames, GLM, CategoricalArrays, FixedEffectModels, CSV

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
