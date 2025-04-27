function simulate_model(para,sols,S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 

    @unpack_Model_Parameters para
    @unpack val_func,a_pol_func,M_pol_func, H_pol_func, a_grids = sols

    # Distribution over the initial permanent component
    initial_dist = Categorical(σ_0_grid)

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ϵ[1,:])

    # State-contingent distributions over the permanent components
    perm_dists = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Outputs
    assets = zeros(S,N+1) # Saving by Age
    cash_on_hand = zeros(S,N+1)
    consumption = zeros(S,N+1) 
    persistent = zeros(S,N+1)
    transitory = zeros(S,N+1)
    housing = zeros(S,N+1)
    mortgage = zeros(S,N+1)
    wealth = zeros(S,N+1) # Savings + Housing

    for s = 1:S
        ϵ_index   = rand(transitory_dist)
        ζ_index = rand(initial_dist)

        # Start with 0 assets, housing and mortgage debt.
        H_index = 1 
        M_index = 1
        a_index = 1

        housing[s,1]  = H_grid[H_index]

        start_ap = findfirst(x -> x >= 0.0, a_grids[:,1])

        assets[s,1]   = a_grids[start_ap, 1]
        mortgage[s,1] = M_grid[M_index]

        # Compute cash on hand 
        P = P_bar 
        cash_on_hand[s,1] = exp(κ[1,2] + ζ_grid[ζ_index] + ϵ_grid[ϵ_index]) + rent_prop * P 

        # Compute choices 
        housing[s,2]  = H_pol_func[H_index,ϵ_index, ζ_index, M_index, a_index,  1]
        assets[s,2]   = a_pol_func[H_index,ϵ_index, ζ_index, M_index, a_index,  1]
        mortgage[s,2] = M_pol_func[H_index,ϵ_index, ζ_index, M_index,a_index,  1]
   
        # Find next period's indices
        H_prime_index = findfirst(x -> x == housing[s,2], H_grid) # Find the index of the previous choice of housing
        M_prime_index = findfirst(x -> x == mortgage[s,2], M_grid) # Find the index of the previous choice of mortgages (it will be on the grid). 
        a_prime_index = findfirst(x -> x == assets[s,2], a_grids[:,2])
        
        # Compute savings 
        wealth[s,1] = assets[s,1] + P * housing[s,2] - mortgage[s,2]

        # Compute consumption 
        consumption[s,1] = budget_constraint(assets[s,2], cash_on_hand[s,1], M_index, M_prime_index, H_index, H_prime_index, P, para )

        if consumption[s,1] < 0
            println("s is: ",s, " n is: ",1, " Consumption is: ", consumption[s,1], "assets are: ", assets[s,1], "asset policy: ", assets[s,2])
        end
        # Save persistent and transitory values
        persistent[s,1] = ζ_grid[ζ_index]
        transitory[s,1] = ϵ_grid[ϵ_index]

        for n = 2:N
            # Draw new values for the labor income shocks 
            ζ_index = rand(perm_dists[ζ_index]) # Draw the new permanent component based upon the old one. 
            ϵ_index = rand(transitory_dist) # Draw the transitory component 

            # Save persistent and transitory values
            persistent[s,n] = ζ_grid[ζ_index]
            transitory[s,n] = ϵ_grid[ϵ_index]

            # Turn the indices of the choices last period to the states today.
            H_index = H_prime_index
            M_index = M_prime_index
            a_index = a_prime_index

            # Compute cash on hand 
            P = P_bar * exp(b * (n-1))
            cash_on_hand[s,n] = exp(κ[n,2] + ζ_grid[ζ_index] + ϵ_grid[ϵ_index]) + (1.0 + R_F) * assets[s,n] + rent_prop * P 

            if a_index == nothing 
                println("s is: ",s, " n is: ",n, " Consumption is: ", consumption[s,n], "assets are: ", assets[s,n], "assets yesterday were ", assets[s,n-1])
            end

            # Compute choices 
            housing[s,n+1]  = H_pol_func[H_index,ϵ_index, ζ_index, M_index, a_index,  n]
            assets[s,n+1]   = a_pol_func[H_index,ϵ_index, ζ_index, M_index, a_index,  n]
            mortgage[s,n+1] = M_pol_func[H_index,ϵ_index, ζ_index, M_index, a_index,  n]
   
            # Find next period's indices
            H_prime_index = findfirst(x -> x == housing[s,n+1], H_grid) # Find the index of the previous choice of housing
            M_prime_index = findfirst(x -> x == mortgage[s,n+1], M_grid) # Find the index of the previous choice of mortgages (it will be on the grid). 
            a_prime_index = findfirst(x -> x == assets[s,n+1], a_grids[:,n+1])

            # Compute savings 
            wealth[s,n] = assets[s,n] + P * housing[s,n+1] - mortgage[s,n]

            # Compute consumption 
            consumption[s,n] = budget_constraint(assets[s,n+1],cash_on_hand[s,n],
                findfirst(x -> x == mortgage[s,n], M_grid), 
                findfirst(x -> x == mortgage[s,n+1], M_grid),
                findfirst(x -> x == housing[s,n], H_grid),
                findfirst(x -> x == housing[s,n+1], H_grid),
                P,
                para
            )

            if consumption[s,n] < 0
                println("s is: ",s, " n is: ",n, " Consumption is: ", consumption[s,n], "assets are: ", assets[s,n])
            end
        

        end 
    end 

    return wealth[:,1:N], assets[:,1:N], consumption[:,1:N], persistent[:,1:N],transitory[:,1:N], cash_on_hand[:,1:N], mortgage[:,1:N], housing[:,1:N]
end
