# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Problem_with_Heloc(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, a_pol_func, M_pol_func, H_pol_func, D_pol_func, a_grids = sols

    # Adds negative savings if you are a homeowner. 
    update_solutions(sols)

    println("Begin solving the model backwards")

    for j in N:-1:1  # Backward induction
        println("Age is ", 24 + j)

        # House prices are deterministic 
        P = P_bar * exp(b * (j-1))

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

                                    # Allow homeowner to take out debt against their home. 
                                    if H_prime == 1 && j < N 
                                        Home_Equity = P * H_prime - M_prime 
                                        start_ap = findfirst(x -> x > -(1-d) * Home_Equity , a_grids[:,j+1])
                                    else 
                                        start_ap = findfirst(x -> x >= 0.0 , a_grids[:,j+1])
                                    end
                                    # Loop over the agent's choice of assets next period: 
                                    for ap_index in start_ap:na
                                        ap = a_grids[ap_index,j+1]
                                        c = budget_constraint(ap, coh, M_index, M_prime_index, H_index, H_prime_index, P, para)
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

function update_solutions(sols::Solutions)

    sols.a_grids = zeros(para.na,para.N + 1)

    for j = 1:para.N + 1
        a_min = -100000
        a_max = 1000000  # Maximum assets on the grid rises
        sols.a_grids[:,j] = collect(range(start = a_min, length = para.na, stop = a_max)) 
    end 
end 



Solve_Problem_with_Heloc(para,sols)

wealth, assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing = simulate_model(para, sols, 10000)