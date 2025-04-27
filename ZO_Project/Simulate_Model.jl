function simulate_model(para,sols,S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 

    @unpack_Model_Parameters para
    @unpack val_func,c_pol_func,M_pol_func, H_pol_func, X_grids = sols

    # Compute policy interpolation functions 

    # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
    c_interp_functions   = Matrix{Any}(undef, 2 * nζ * nM, N) 
    M_interp_functions   = Matrix{Any}(undef, 2 * nζ * nM, N) 
    H_interp_functions   = Matrix{Any}(undef, 2 * nζ * nM, N) 

    for j = 1:N
        for M_index in 1:nM
            for ζ_index in 1:nζ
                for H_index in 1:nH

                    # Compute linear index 
                    index = (H_index - 1) * (nM * nζ) + (M_index - 1) * nζ + (ζ_index - 1) + 1
    
                    # Access next year's with dimensions [H_index, ζ, H, X, j+1]
                    c_interp_functions[index,j]   = linear_interp(c_pol_func[H_index, ζ_index, M_index, :, j], X_grids[:,j])
                    M_interp_functions[index,j]   = linear_interp(M_pol_func[H_index, ζ_index, M_index, :, j], X_grids[:,j])
                    H_interp_functions[index,j]   = linear_interp(H_pol_func[H_index, ζ_index, M_index, :, j], X_grids[:,j])
                end
            end
        end
    end 

    # Distribution over the initial permanent component
    initial_dist = Categorical(σ_0_grid)

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ϵ[1,:])

    # State-contingent distributions over the permanent components
    perm_dists = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Outputs
    assets = zeros(S,N) # Saving by Age
    cash_on_hand = zeros(S,N)
    consumption = zeros(S,N) 
    persistent = zeros(S,N)
    transitory = zeros(S,N)
    housing = zeros(S,N)
    mortgage = zeros(S,N)

    for s = 1:S
        ϵ_index   = rand(transitory_dist)
        ζ_index = rand(initial_dist)

        # Start with 0 assets, housing and mortgage debt.
        H_index = 1 
        M_index = 1
        index = (H_index - 1) * (nM * nζ) + (M_index - 1) * nζ + (ζ_index - 1) + 1

        # Compute cash on hand 
        cash_on_hand[s,1] = exp(κ[1,2] + ζ_grid[ζ_index] + ϵ_grid[ϵ_index])

        # Compute choices (reassigning cash on hand to the nearest grid value if it is in a discontinuity in the H policy function) 
        cash_on_hand[s,1], housing[s,1], consumption[s,1], mortgage[s,1] = compute_choices(cash_on_hand[s,1], index, 1, H_interp_functions,c_interp_functions,M_interp_functions, para, sols)        
        
        # Compute other variables needed to find cash on hand tomorrow 
        P = P_bar 
        H_index_next = findfirst(x -> x == housing[s,1], H_grid) # Find the index of the previous choice of housing
        M_index_next = findfirst(x -> x == mortgage[s,1], M_grid) # Find the index of the previous choice of mortgages (it will be on the grid). 

        # Compute savings 
        assets[s,1] = budget_constraint(consumption[s,1], cash_on_hand[s,1], M_index, M_index_next,H_index, H_index_next, P, para)

        # Save persistent and transitory values
        persistent[s,1] = ζ_grid[ζ_index]
        transitory[s,1] = ϵ_grid[ϵ_index]

        for n = 2:35 
            # Draw new values for the labor income shocks 
            ζ_index = rand(perm_dists[ζ_index]) # Draw the new permanent component based upon the old one. 
            ϵ_index = rand(transitory_dist) # Draw the transitory component 

            # Turn the indices of the choices last period to the states today.
            H_index = H_index_next
            M_index = M_index_next

            index = (H_index - 1) * (nM * nζ) + (M_index - 1) * nζ + (ζ_index - 1) + 1
            
            # Compute cash on hand 
            cash_on_hand[s,n] = exp(κ[1,2] + ζ_grid[ζ_index] + ϵ_grid[ϵ_index]) + R_F * assets[s,n-1]

            # Compute choices (reassigning cash on hand to deal with discontinuities in the H policy function)

            cash_on_hand[s,n], housing[s,n], consumption[s,n], mortgage[s,n] = compute_choices(cash_on_hand[s,n],ζ_index, M_index, H_index, n, H_interp_functions,c_interp_functions,M_interp_functions,para, sols)        
        
            # Compute other variables needed to find cash on hand tomorrow 
            P = P_bar * exp(b * (n-1))
            H_index_next = findfirst(x -> x == housing[s,n], H_grid) # Find the index of the previous choice of housing
            M_index_next = findfirst(x -> x == mortgage[s,n], M_grid) # Find the index of the previous choice of mortgages (it will be on the grid). 

            println("index is: ",index, " s = ", s, " n = ", n," ζ_index = ", ζ_index, " H_index = ", H_index, " H_index_next = ", H_index_next," M_index = ", 
            M_index, " M_index_next = ", M_index_next, " Housing = ", housing[s,n]," Mortgage = ", mortgage[s,n],
            " cash_on_hand is: ", cash_on_hand[s,n])
            # Compute savings 
            assets[s,n] = budget_constraint(consumption[s,n], cash_on_hand[s,n], M_index, M_index_next, H_index, H_index_next, P, para)

            # Save persistent and transitory values
            persistent[s,n] = ζ_grid[ζ_index]
            transitory[s,n] = ϵ_grid[ϵ_index]

        end 
    end 

    return assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing
end

function compute_choices(cash_on_hand::Float64,ζ_index::Int64,M_index::Int64,H_index::Int64, n::Int64, H_interp_functions::Matrix{Any},c_interp_functions::Matrix{Any},M_interp_functions::Matrix{Any},para::Model_Parameters, sols::Solutions)
    @unpack X_grids = sols
    @unpack_Model_Parameters para

    # Compute the choices at that state 
    housing    = H_interp_functions[index,n](cash_on_hand)
    consumption = c_interp_functions[index,n](cash_on_hand)
    mortgage   = M_interp_functions[index,n](cash_on_hand)

    # If Housing is in a discontinuity, just associate that observation to the nearest 
    # point on the grid. 
    if findfirst(x -> x == housing, H_grid) == nothing || findfirst(x -> x == mortgage, M_grid) == nothing
        # For now, just go with the rule for the closest point on the grid to X. 
        X_high = findfirst(x -> x > cash_on_hand, X_grids[:,n])
        X_low = X_high - 1
        if abs(X_grids[X_high,n] - cash_on_hand) < abs(X_grids[X_low,n] - cash_on_hand)
            X_interp = X_high
        else 
            X_interp = X_low
        end 
        # Make decision based on closest point's policy rule. 
        housing     = H_pol_func[H_index, ζ_index, M_index, X_interp,  n]
        consumption = c_pol_func[H_index, ζ_index, M_index,X_interp, n]
        mortgage    = M_pol_func[H_index, ζ_index, M_index,X_interp, n]
        cash_on_hand = X_interp
    end 
   return cash_on_hand, housing, consumption, mortgage
end 

simulate_model(para, sols, 1000)

# Check why the saving policy seems to be so low 
a_pol_func = zeros(2, nζ, nM, nX, N)

for j = 1:N
    P = P_bar * exp(b * (j-1))
    for X in 1:nX
        for ζ in 1:nζ
            for m in 1:nM
                for h in 1:2  # assuming 2 types (rent/buy)
                    M_prime_index = findfirst(i -> M_grid[i] == M_pol_func[h,ζ,m,X,j], eachindex(M_grid))
                    H_prime_index = findfirst(i -> H_grid[i] == H_pol_func[h,ζ,m,X,j], eachindex(H_grid))
                    a_pol_func[h,ζ,m,X,j] = budget_constraint(
                        c_pol_func[h,ζ,m,X,j],
                        X_grids[X,j],
                        m,
                        M_prime_index,
                        h,
                        H_prime_index,
                        P,
                        para
                    )
                end
            end
        end
    end
end

plot(X_grids[:,1],c_pol_func[2,3,9,:,10])

H_pol_func[2,3,9,:,10]


