# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, a_pol_func, M_pol_func, H_pol_func, rent_prop, P_bar, a_grids = sols

    println("Begin solving the model backwards")

    for j in N:-1:1  # Backward induction
        println("Age is ", 24 + j)

        # House prices are deterministic 
        P = P_bar * exp(b * (j-1))

    #= I don't need to interpolate

       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
       interp_functions = Vector{Any}(undef, 2 * nζ * nM) 
       for M_index in 1:nM
            for ζ_index in 1:nζ
                for H_index in 1:nH

                    # Compute linear index 
                    index = (H_index - 1) * (nM * nζ) + (M_index - 1) * nζ + (ζ_index - 1) + 1

                    # Access next year's with dimensions [H_index, ζ, H, X, j+1]
                    interp_functions[index] = linear_interp(val_func[H_index, ζ_index, M_index, :, j+1], X_grids[:,j+1])
                end
           end
       end

    =#

        # Loop over Housing States 
        Threads.@threads for a_index in 1:na
            a = a_grids[a_index,j]

            # Loop over Mortgage Debt states
            for M_index in 1:nM
                M = M_grid[M_index]

                # Loop over persistent income states
                for ζ_index in 1:nζ
                    ζ = ζ_grid[ζ_index]

                    # Loop over transitory income states
                    for ϵ_index in 1:nϵ
                        ϵ = ϵ_grid[ϵ_index]

                        Y = exp(ϵ + ζ + κ[j,2])
                        # Loop over whether the agent chooses to buy a house or not
                        for H_index in 1:nH
                            H = H_grid[H_index]

                            candidate_max = pun 
                        
                            # Loop over the agent's choice of mortgage 
                            for M_prime_index in 1:nM
                                M_prime = M_grid[M_prime_index]

                                # Loop over the agent's choice of housing: 
                                for H_prime_index in 1:nH 
                                    H_prime = H_grid[H_prime_index]

                                    # If the mortgage choice does not satisfy the constraint, skip. 
                                    if mortgage_constraint(M_index, M_prime_index, H_index, H_prime_index, P, j, para) < 0 
                                        continue 
                                    end 

                                    coh =  (1 + R_F) * a + Y + rent_prop * P 

                                    # Loop over the agent's choice of assets next period: 
                                    for ap_index in 1:na
                                        ap = a_grids[ap_index,j+1]

                                        c = budget_constraint(ap, coh, M_index, M_prime_index, H_index, H_prime_index, P, para, rent_prop)
                                        # Feasibility constraint
                                        if c > 0 
                                            val = flow_utility_func(c, H_index, para, sols)
                                            for ϵ_prime_index in 1:nϵ
                                                for ζ_prime_index in 1:nζ
                                                    val += β * T_ϵ[ϵ_index,ϵ_prime_index] * T_ζ[ζ_index,ζ_prime_index] * val_func[H_prime_index, ϵ_prime_index, ζ_prime_index, M_prime_index, ap_index, j + 1]
                                                end
                                            end

                                            if val > candidate_max  # Check for max
                                                val_func[ H_index,ϵ_index, ζ_index, M_index, a_index, j]      = val

                                                a_pol_func[ H_index,ϵ_index, ζ_index, M_index, a_index, j]    = ap 
                                                H_pol_func[ H_index,ϵ_index, ζ_index, M_index, a_index, j]    = H_prime
                                                M_pol_func[ H_index,ϵ_index, ζ_index, M_index, a_index, j]    = M_prime
            
                                                candidate_max = val 
                                            end 
                                        end 
                                    end # a_prime Loop
                            
                                end # H_prime Loop
                            end # M_prime Loop

                            # If no choice can possibly satisfy the constraints, then return the punishment utility 
                            if candidate_max == pun 
                                val_func[ H_index, ϵ_index, ζ_index, M_index, a_index, j] = pun 
                            end 
                        end # H Loop

                    end # ϵ Loop
                end # ζ Loop
            end # M Loop
        end # X Loop
    end # T loop
end

######################################
# Optimization Functions 
######################################
# Compute expectation function given both c and D
function compute_worker_value(c::Float64, j::Int64, X::Float64, M_index::Int64, M_prime_index::Int64, H_index::Int64, H_prime_index::Int64,
                            P::Float64, ζ_index::Int64, interp_functions::Vector{Any}, para::Model_Parameters, sols::Solutions )

    @unpack_Model_Parameters para

    # Compute savings today from what is left over after H_prime, c and M_prime are chosen. 
    A   = budget_constraint( c, X, M_index, M_prime_index, H_index, H_prime_index, P, para,sols)
    
    if A < 0 
        return pun 
    else

        val = flow_utility_func(c, H_index, para, sols)

        # Find the continuation value 
        # Loop over random variables 
        for ζ_prime_index in eachindex(ζ_grid)
            ζ_prime = ζ_grid[ζ_prime_index]
            
            for ϵ_prime_index in eachindex(ϵ_grid)
                ϵ_prime = ϵ_grid[ϵ_prime_index]

                if j == N 
                    Y_Prime = 0.0 
                else

                    Y_Prime = exp(κ[j+1, 2]  + ζ_prime + ϵ_prime)
                end 

                # Compute next period's liquid wealth
                X_prime =  (1 + R_F) * A + Y_Prime 
                                                                    
                    val += β * (  T_ζ[ζ_index, ζ_prime_index]  * T_ϵ[1, ϵ_prime_index] *
                            interp_functions[(H_prime_index - 1) * (nM * nζ) + (M_prime_index - 1) * nζ + (ζ_prime_index - 1) + 1](X_prime)
                                )    
            
            end 
        end 
        return val 
    end 
end 

# Optimize value function over c given a choice of D
function optimize_worker_c(j::Int64, X_index::Int64, M_index::Int64, M_prime_index::Int64, H_index::Int64, H_prime_index::Int64, 
                            ζ_index::Int64, P::Float64,interp_functions::Vector{Any}, para::Model_Parameters, sols::Solutions)

    @unpack_Model_Parameters para
    @unpack X_grids = sols

    X = X_grids[X_index, j]

    # Find maximum feasible consumption
    budget(c) =  budget_constraint(c, X, M_index, M_prime_index, H_index, H_prime_index, P, para)

    c_max_case = find_zero(budget, X, Roots.Order1())

    if c_max_case < 0
        return (c_opt = 0.0, val_opt = pun)

    else
        # Optimize using Brent's method
        result = optimize(c -> -compute_worker_value(c, j, X, M_index, M_prime_index, H_index, H_prime_index, P, ζ_index, interp_functions, para, sols), 0.0, c_max_case, Brent())
        c_opt = Optim.minimizer(result)

        val_opt = compute_worker_value(c_opt, j, X, M_index, M_prime_index, H_index, H_prime_index, P, ζ_index, interp_functions, para, sols)
        return (c_opt, val_opt)
    end 
end