# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, M_pol_func, H_pol_func= sols

    println("Begin solving the model backwards")

    for j in N:-1:1  # Backward induction
        println("Age is ", 20 + j)

        # House prices are deterministic 
        P = P_bar * exp(b * (j-1))

       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
       interp_functions = Vector{Any}(undef, 2 * nζ * nM) 
       for H_index in 1:2
            for M_index in 1:nM
               for ζ_index in 1:nζ
                       # Compute linear index 
                       index =  (H_index - 1) * (ζ_index * 4) + (ζ_index - 1) * 4 + (Inv_Move_index - 1) * 2 + (IFC_index - 1) + 1
                        # Access val_func with dimensions [Inv_Move, IFC, ζ, H, X, j]
                       interp_functions[index] = linear_interp(val_func[Inv_Move_index, IFC_index, ζ_index, H_index, :, j+1], X_grid)
                   end
           end
       end

        # Loop over Housing States 
        Threads.@threads for X_index in 1:nX
            X = X_grid[X_index]

            # Loop over Mortgage Debt states
            for M_index in 1:nM
                M = M_grid[M_index]

                # Loop over persistent income states
                for ζ_index in 1:nζ

                    # Loop over whether the agent chooses to buy a house or not
                    for H_index in 1:2
                        H = H_grid[H_index]

                        candidate_max = pun 
                        
                        # Loop over the agent's choice of mortgage 
                        for M_prime_index in 1:nM
                            M = M_grid[M_prime_index]

                            # Loop over the agent's choice of housing: 
                            for H_prime_index in 1:nH 
                                H_prime = H_grid[H_prime_index]

                                # If the mortgage choice does not satisfy the constraint, skip. 
                                if mortgage_constraint(M, M_prime, H, H_prime,P, j, para) < 0 
                                    continue 
                                end 

                                # Given the other choices, find the agent's optimal consumption choice: 
                                c   = optimize_worker_c(j, X, M, H, P, ζ_index, H_prime, interp_functions, para)
                                val = compute_worker_value(c, j, X, M, ζ_index, H, P, H_prime, interp_functions, para)

                                 # Update value function
                                 if val > candidate_max 
                                    val_func[ H_index, ζ_index, M_index, X_index, j]     = val

                                    c_pol_func[ H_index, ζ_index, M_index, X_index, j]    = c 
                                    H_pol_func[ H_index, ζ_index, M_index, X_index, j]    = H_prime
                                    M_pol_func[ H_index, ζ_index, M_index, X_index, j]    = M_prime

                                    candidate_max = val 
                              
                                end 
                            end # H_prime Loop
                        end # M_prime Loop

                        # If no choice can possibly satisfy the constraints, then return the punishment utility 
                        if candidate_max == pun 
                            val_func[ H_index, ζ_index, M_index, X_index, j] = pun 
                        end 

                    end # ζ Loop
                end # H Loop
            end # M Loop
        end # X Loop
    end # T loop
end

######################################
# Optimization Functions 
######################################
# Compute expectation function given both c and D
function compute_worker_value(c::Float64, j::Int64,X::Float64,M::Float64, ζ_index::Int64, H::Float64, 
                            P::Float64, H_prime::Float64, interp_functions::Vector{Any},para::Model_Parameters )

    @unpack_Model_Parameters para

    # Compute bonds that the agentis buying
    B = budget_constraint(c, X, M, H, P, H_prime, M_prime, para)

    val = flow_utility_func(c, H, para)

    # Find the continuation value 
    # Loop over random variables 
    for ζ_prime_index in eachindex(ζ_grid)
        ζ_prime = ζ_grid[ζ_prime_index]
        
        for ϵ_prime_index in eachindex(ϵ_grid)
            ϵ_prime = ϵ_grid[ϵ_prime_index]

            Y_Prime = exp(κ[j+1, 2]  + ζ_prime + ϵ_prime)

            # Compute next period's liquid wealth
            X_prime =  R_F * B + Y_Prime
                                                                
                val += β * (  T_ζ[ζ_index, ζ_prime_index]  * T_ω[1, ω_prime_index] *
                        interp_functions[(H_prime_index - 1) * (ζ_prime_index * 4) + (ζ_prime_index - 1) * 4 + (1 - 1) * 2 + (IFC_prime_index - 1) + 1](X_prime) +
                        T_ζ[ζ_index, ζ_prime_index]   * T_ω[1, ω_prime_index] *
                        interp_functions[(H_prime_index - 1) * (ζ_prime_index * 4) + (ζ_prime_index - 1) * 4 + (2 - 1) * 2 + (IFC_prime_index - 1) + 1](X_prime)   
                        )    
        
        end 
    end 
    return val 
end 

# Optimize value function over c given a choice of D
function optimize_worker_c(j::Int64, X::Float64,M::Float64, H::Float64, P::Float64, ζ_index::Int64, H_prime::Float64, interp_functions::Vector{Any}, para::Model_Parameters)
    @unpack_Model_Parameters para

    # Find maximum feasible consumption
    budget(c) =  budget_constraint(c, X, M, H, P, H_prime, M_prime, para)

    c_max_case = find_zero(budget, X, Roots.Order1())

    if c_max_case < 0
        return (c_opt = 0.0)

    else
        # Optimize using Brent's method
        result = optimize(c -> -compute_worker_value(c, j, X, M, H, P, ζ_index, H_prime, interp_functions, para), 0.0, c_max_case, Brent())
        
        return (c_opt = Optim.minimizer(result))
    end 
end

#=
# Find the value of the problem given a choice of D taking the optimizing choice of c function as given. 
function objective_worker_D(D::Float64, j::Int64, H::Float64, P::Float64, X::Float64, ζ_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, 
                        H_prime_index::Int64,IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters)
    @unpack_Model_Parameters para

    # Get optimal c for this D
    c = optimize_worker_c(j, H, P, X, ζ_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
    
    # Compute value
    val = compute_worker_value(j, H, P,  X, ζ_index, Inv_Move, c, α, H_prime, H_prime_index, D,IFC, FC, interp_functions, para)
    return val  # Negative for maximization
end

function optimize_worker_d(j::Int64, H::Float64, P::Float64, X::Float64, ζ_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, H_prime_index::Int64,IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters)
    @unpack_Model_Parameters para
    # Find maximum feasible consumption
    debt_limit(D) =  debt_constraint(D, H_prime, P, para)
    D_max_case = find_zero(debt_limit, X, Roots.Order1())

    if D_max_case < 0
        return (D_opt = 0.0, val_opt = pun)
    else
    
    # Optimize using Brent's method
    result = optimize(D -> -objective_worker_D(D, j, H, P,  X, ζ_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, para), 0.0, D_max_case, Brent())
    D_opt = Optim.minimizer(result)

    c_opt = optimize_worker_c(j, H, P, X, ζ_index, Inv_Move,α, H_prime, H_prime_index, D_opt, IFC, FC, interp_functions, para)
    val_opt = -Optim.minimum(result)
        return (D_opt, c_opt, val_opt)
    end 
end
=#