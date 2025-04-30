# Finds the value of housing services from owning which
# matches homeownership rates at age 45 in the model 
# to homeownership rates at age 45 in the SCF + over 1970 - 1998. 
# data moment = 79.1%.  
function calibrate_model(data_moment::Float64, para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para
    @unpack val_func,a_pol_func,M_pol_func, H_pol_func, a_grids, s = sols

    error = 1.0 

    tol = 10^-2 
    s_guess = 1.0 

    n = 1
    while error > tol 

        sols.s = s_guess 

        Solve_Problem(para,sols) 
        wealth, assets, consumption, persistent,transitory, cash_on_hand, mortgage, housing = simulate_model(para, sols, 5000)

        # The moment I want to match is the homeownership rate at age 45. 
        model_moment = mean(housing[:,20])

        error = abs(data_moment - model_moment)

        # Update s_guess in the same direction as the miss
        s_guess = s_guess +  0.5 * (data_moment - model_moment)/model_moment
        n += 1
        println("Model Moment is: ", model_moment, " n is: ", n, " s guess is : ", s_guess)
    end 

    println("Final s is: ",sols.s)
end 


