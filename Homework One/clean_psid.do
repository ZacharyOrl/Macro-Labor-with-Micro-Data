set logtype text
set more off
****************************************************************
* Econ 810: Spring 2025 Advanced Macroeconomics 
* by J. Carter Braxton 
* Persistent & Transitory Income in the Bewley model
* Authors: Zachary Orlando and Cutberto Frias Sarraf
****************************************************************
* Set earnings cutoff requirements 

local cutoff = 3500 // Close to the 1% level for the sample. 
local minimimum_times_satisified = 5
****************************************************************

local current_dir : pwd

* Set input location 
global indir "`current_dir'"
* Set output location 
global outdir_parameters "$indir/parameters"
global outdir_images "$indir/images"
****************************************************************
* Variables Used: 
* Key: 
* x11101LL = Individual identifier 
* year = year of survey

* Income: 
* i11113 = HH Post-Government income (TAXSIM so post 1991 years are not dropped )

* Demographics (for Deterministic Component):
* d11102LL = Gender 
* d11101 = Age 
* d11112LL = Race 
* d11106 = Number of Persons in Household 
* d11107 = Number of Children in Household
* d11109 = Number of years of education
* e11104 = working satus (0 = working, 1 = not)

* Additional vars used for sample restrictions: 
* x11104LL = Oversample identifier (11 = Main Sample, 12 = SEO)
* d11105 = Relationship to Household Head (Head = 1,Partner = 2, Child = 3, Relative = 4, Non-Relative = 5)
* w11102 = Household Weight
****************************************************************
cd "$indir"

* Load data 
use "pequiv_long.dta",clear

* Rename variables
local oldnames x11101LL i11113 d11102LL d11101 d11112LL d11106 d11107 x11104LL d11105 d11109 e11104 w11102
local newnames person_id hh_income  gender age race hh_size hh_children oversample_id relationship_head education_years working_status wgt


local i = 1
foreach oldvar in `oldnames' {
    local newvar: word `i' of `newnames'
    rename `oldvar' `newvar'
    local ++i
}

keep person_id year hh_income gender age race hh_size hh_children oversample_id relationship_head education_years working_status wgt

* Income is reported in nominal dollars, adjust for inflation using annual price index for all urban consumers in US. 
preserve 

	clear all 
	
	import delimited "CPI.csv"
	
	gen year = yofd(date(observation_date, "MDY"))
	drop observation_date 
	
	rename cpiaucsl price_level
	tempfile infl
	
	save `infl'

restore 

merge m:1 year using `infl'
drop if _merge == 2
drop _merge 

replace hh_income = 100* hh_income/price_level

* Impose sample restrictions 
keep if inrange(year,1978,1997) // year must be between 1978 and 1997 
keep if oversample_id == 11 // Drop the SEO sample 
keep if relationship_head == 1 // Consider household heads only 
keep if inrange(age, 25,59)	// Consider only observations where the head was working-age

* Drop if demographics are missing
gen missing_dem = 0
foreach var in `newnames' {
	
	replace missing_dem = 1 if missing(`var')
}

drop if missing_dem == 1 

drop missing_dem 	

* Impose that households must have income exceeding the cutoff in at least `minimimum_times_satisified' years
gen qualif_inc = (hh_income >= `cutoff' ) // Did income this year exceed the cutoff? 
bysort person_id (year): egen num_qual_inc = total(qualif_inc) // Number of years the HH's income exceeded the cutoff 
keep if num_qual_inc >= `minimimum_times_satisified'

drop num_qual_inc 
* Drop years where HH income < cutoff. 
keep if qualif_inc == 1

drop qualif_inc

* Construct the cohort 
gen cohort = year - age

* Partial out deterministic income component 
gen log_hh_inc = log(hh_income)
reghdfe log_hh_inc i.age i.cohort [pweight = wgt]
predict deterministic_component

* Age Profile of Predictable Income graph
preserve 
	
	collapse (mean) deterministic_component [pweight = wgt], by(age)
	
	graph twoway (line det* age, sort lcolor(black) lwidth(thick)), ///
    ytitle("Log After-Tax Income" "(1997 $)", size(small)) ///
    xtitle("Age") title("Age Profile of Income") ///
    xlabel(25(5)60) xscale(range(25 60))
	
	 graph export "$outdir_images/age_profile_psid.png", width(2000) replace
	 
	 keep age deterministic_component
	 
	 sort age 
	 
	 export delimited "$outdir_parameters/life_cycle_income.csv",replace 
restore 

* Generate the persistent + transitory income process 
gen residual_component = log_hh_inc - deterministic_component

sort person_id year
drop oversample_id relationship_head 

****************************************************************
* Construct the table of descriptive statistics 
****************************************************************
cd "$outdir_parameters"

// Create race indicators
gen white = (race == 1)
gen black = (race == 2)

rename working_status working
replace working = 2 - working
replace gender = gender - 1

* Define variables to summarize
local vars "hh_income age gender education_years working white black hh_size hh_children"

* Prepare matrix to store results
levelsof year, local(years)
local nyears : word count `years'
local nvars : word count `vars'

* Create matrix to store results
mata: results = J(`nyears', `nvars' + 1, .)

* Prepare column and row names
local results_cols "`vars' N"
mata: st_local("results_cols", "`: local results_cols'")
mata: st_local("results_rows", "`: local years'")

* Prepare to store formatted results
local formatted_results ""

* Store summary statistics
local i 1
foreach y of local years {
    local row_results ""
    local j 1
    foreach v of local vars {
        quietly summarize `v' if year == `y'
        local mean = r(mean)
        local sd = r(sd)
        local n = r(N)
        
        * Format mean and SD using string()
        local formatted_stat = string(`mean', "%9.2f") + "\\" + "(" + string(`sd', "%9.2f") + ")"
        
        * Store in Mata matrix
        mata: results[`i', `j'] = `mean'
        
        * Collect for display/export
        local row_results `"`row_results' `"`formatted_stat'"'"'
        local j = `j' + 1
    }
    
    * Add N to the row
    local row_results `"`row_results' `n'"'
    local formatted_results `"`formatted_results' `"`row_results'"'"'
    
    local i = `i' + 1
}

* Create better variable labels
local varlabels ""
foreach v of local vars {
    local varlabel : variable label `v'
    if "`varlabel'" == "" local varlabel "`v'"
    local varlabels `"`varlabels' `v' "`varlabel'""'
}
local varlabels `"`varlabels' N "Observations""'

* Export to LaTeX
file open myfile using "descriptives_by_year.tex", write replace
file write myfile "\begin{table}[htbp]" _n
file write myfile "\centering" _n
file write myfile "\caption{Descriptive Statistics by Year}" _n
file write myfile "\begin{tabular}{l*{`nvars'}{c}c}" _n
file write myfile "\hline" _n

* Write headers
file write myfile "Year"
foreach v of local vars {
    local varlabel : variable label `v'
    if "`varlabel'" == "" local varlabel "`v'"
    file write myfile " & `varlabel'"
}
file write myfile " & Observations \\" _n
file write myfile "\hline" _n

* Write data rows
local i 1
foreach y of local years {
    file write myfile "`y'"
    local row : word `i' of `formatted_results'
    foreach stat of local row {
        file write myfile " & `stat'"
    }
    file write myfile " \\" _n
    local i = `i' + 1
}

file write myfile "\hline" _n
file write myfile "\end{tabular}" _n
file write myfile "\end{table}" _n
file close myfile

di "Descriptive statistics exported to descriptives_by_year.tex"
****************************************************************
* Save cleaned data for analysis. 
export delimited "psid_cleaned.csv",replace
clear all
exit
