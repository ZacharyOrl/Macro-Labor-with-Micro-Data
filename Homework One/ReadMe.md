## Code for BPP, Kaplan-Violante & the Bewley Model

This code base:  
1. Estimates predictable (based on observables) lifecycle household income from the PSID (1978-1997) and a residual component.  
2. Estimates the variance of persistent and transitory shocks using the residual component.  
3. Discretizes these processes for input into a lifecycle Bewley model.  
4. Solves and simulates the Bewley model and examines the accuracy of the BPP method of measuring households' levels of insurance against persistent and transitory income shocks.  

Refer to the file **"master.jl"**, which runs all the programs.  
The results are in **Results_Writeup.pdf**.

### Requirements:
To run the code, you need:  
1. The file **"pequiv_long.dta"** – the Cross-National Equivalent File 1970-2015 for the PSID (not included in the folder).  
2. Annual data on price levels – we use CPI for All Urban Consumers in the US.  
3. Stata installed (for the first script).  
4. The relevant Julia packages called by each script.   

**Zachary and Cutberto**
