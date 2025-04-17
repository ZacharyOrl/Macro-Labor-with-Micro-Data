set logtype text
set more off
*******************************************************************************
* Econ 810: Spring 2025 Advanced Macroeconomics 
* by J. Carter Braxton 
* Author: Zachary Orlando and Cutberto Frias Sarraf
*******************************************************************************

*******************************************************************************
* Set working directory (change to your directory)
cd "\\sscwin\dfsroot\labusers\friassarraf\Documents\STATA\810_ECON_Carter\Homework Four"
pwd   // Check the current working directory
dir   // List files in the directory

local current_dir : pwd
* Set input location 
global indir "`current_dir'"
* Set output location 
global outdir "$indir/output"
*******************************************************************************

*******************************************************************************
* Variables Used: 
* Key: 
* x11101LL 	= Individual identifier 
* x11102 	= Household Identification Number
* year 		= year of survey

* Income: 
* i11113 = HH Post-Government income (TAXSIM so post 1991 years are not dropped )

* Demographics (for Deterministic Component):
* d11102LL 	= Gender 
* d11101 	= Age 
* d11112LL 	= Race 
* d11106 	= Number of Persons in Household 
* d11107 	= Number of Children in Household
* d11109 	= Number of years of education
* e11104 	= Working satus (0 = working, 1 = not)

* Additional vars used for sample restrictions: 
* x11104LL 	= Oversample identifier (11 = Main Sample, 12 = SEO)
* d11105 	= Relationship to Household Head (Head = 1,Partner = 2, Child = 3, Relative = 4, Non-Relative = 5)

*******************************************************************************

cd "$indir"

* Load data 
use "pequiv_long.dta",clear

* Rename variables
local oldnames x11101LL x11102 i11113 d11102LL d11101 d11112LL d11106 d11107 x11104LL d11105 d11109 e11104 e11101
local newnames person_id hh_id hh_income gender age race hh_size hh_children oversample_id relationship_head education_years working_status workings_hours


local i = 1
foreach oldvar in `oldnames' {
    local newvar: word `i' of `newnames'
    rename `oldvar' `newvar'
    local ++i
}

keep person_id year hh_id hh_income gender age race hh_size hh_children oversample_id relationship_head education_years working_status workings_hours

* Income is reported in nominal dollars, adjust for inflation using annual price index for all urban consumers in US. 
preserve 

	clear all 
	
	import delimited "CPIAUCSL_1968.csv"
	
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

*******************************************************************************
* In this section we impose sample restrictions 


keep if inrange(year, 1978, 1997) // Year must be between 1978 and 1997 
keep if oversample_id == 11		  // Drop the SEO sample 
keep if relationship_head == 1 	  // Consider household heads only 
keep if gender			== 1      // Consider males only
keep if inrange(age, 23, 60)	  // Consider only observations where the head was working-age

* Drop if demographics are missing
gen missing_dem = 0
foreach var in `newnames' {
	
	replace missing_dem = 1 if missing(`var')
}

drop if missing_dem == 1 
drop missing_dem

* Construct the cohort 
gen cohort = year - age

sort person_id year

*******************************************************************************
* In this seciton we impose aditional restrictions based on HGA (2011)

* Declare the panel structure
tsset person_id year

* Create a new variable in_sample and initialize it with missing values
gen in_sample = .

* Set in_sample = 1 for males 30 < age if
* 1) The household head is working (working_status == 1) 
* 2) Working hours are satisfy: 520 < h < 5820
* 3) Earn at least $1500 in 1968 dollars
replace in_sample = 1 if 30 < age & working_status == 1 & (520 <= workings_hours <= 5820) & 1500 <= hh_income

* Set in_sample = 1 for males age <= 30 of
* 1) The household head is working (working_status == 1) 
* 2) Working hours are satisfy: 260 < h < 5820
* 3) Earn at least $1000 in 1968 dollars
replace in_sample = 1 if age <= 30 & working_status == 1 & (260 <= workings_hours <= 5820) & 1000 <= hh_income

* We are going to use the observations that satisfy the previous criteria
keep if in_sample == 1
drop price_level in_sample

gen log_hh_income = log(hh_income)

save "PS4_Clean.dta", replace

*******************************************************************************
* In this section, we obtain the stats by age bin j at time t. 
use "PS4_Clean.dta",clear

* Define the tempfile for storing results
tempfile results
save `results', emptyok replace

* Get the list of years in your data
levelsof year, local(years)

forvalues j = 23(1)60 {
	foreach t of local years {
		preserve
			* Restrict to year t and age bin
			keep if year == `t' & age >= `=`j'-2' & age <= `=`j'+2'

			* Skip if no observations
			count
			if r(N) == 0 {
				restore
				continue
			}

			* Calculate mean, var, median			
			quietly summarize log_hh_income, detail
			local mean_income      = r(mean)
			local var_log_income   = r(Var)
			local median_income    = r(p50)
			local sd_log_income    = r(sd)
			local kurtosis_income  = r(kurtosis)

			* Build one-row dataset
			clear
			set obs 1
			gen age_center      	= `j'
			gen year            	= `t'
			gen mean_log_income  	= `mean_income'
			gen var_log_income   	= `var_log_income'
			gen sd_log_income		= `sd_log_income'
			gen kurtosis_log_income	= `kurtosis_income'
			gen sk_log_income    	= `mean_income' / `median_income'

			* Append and save
			append using `results'
			save `results', replace
		restore
	}
}
* Load the final output
use `results', clear
keep if  age_center != .
keep age_center year mean_log_income var_log_income sd_log_income kurtosis_log_income sk_log_income
// gen log_mean_hh_income = log(mean_hh_income)
gsort year age_center
save "PS4_Clean_Moments.dta", replace

*******************************************************************************
* In this section, we estimate the Figure 1: Standard deviation of log earnings
use "PS4_Clean_Moments.dta", clear

* Regression
reghdfe sd_log_income i.year i.age_center, noconstant
parmest, saving("$outdir/Figure_1_Coefficients.dta", replace)

use "$outdir/Figure_1_Coefficients.dta", clear
* Keep only age_center coefficients
keep if strpos(parm, "age_center") > 0

* Extract the age value from the parm string
gen age = real(regexs(1)) if regexm(parm, "([0-9]+)\.age_center")
* Scale all estimates so that the estimate at age 40 is equal to 0.499912
gen scale_factor = .
quietly summarize estimate if age == 40
local scale = r(mean) 
replace scale_factor = estimate - `scale' + 0.499912

* Panel B: Variance of log earnings
graph twoway (line scale_factor age, sort lcolor(black) lwidth(thick)), ///
ytitle("", size(small)) ///
xtitle("Age") title("Figure 1: Standard deviation of log earnings") ///
xlabel(25(5)60) xscale(range(25 60))

graph export "$outdir/Figure_1.png", width(2000) replace
export delimited "$outdir/Figure_1.csv",replace 

*******************************************************************************
* In this section, we estimate the Figure 2: Skewness of log earnings
use "PS4_Clean_Moments.dta", clear

* Regression
reghdfe sk_log_income i.age_center i.year, noconstant
parmest, saving("$outdir/Figure_2_Coefficients.dta", replace)

use "$outdir/Figure_2_Coefficients.dta", clear
* Keep only age_center coefficients
keep if strpos(parm, "age_center") > 0

* Extract the age value from the parm string
gen age = real(regexs(1)) if regexm(parm, "([0-9]+)\.age_center")
* Scale all estimates so that the estimate at age 40 is equal to 0.9969335 
gen scale_factor = .
quietly summarize estimate if age == 40
local scale = r(mean) 
replace scale_factor = estimate - `scale' + 0.9969335

* Figure 2: Skewness of log earnings
graph twoway (line scale_factor age, sort lcolor(black) lwidth(thick)), ///
ytitle("", size(small)) ///
xtitle("Age") title("Figure 2: Skewness of log earnings") ///
xlabel(25(5)60) xscale(range(25 60))

graph export "$outdir/Figure_2.png", width(2000) replace
export delimited "$outdir/Figure_2.csv",replace

*******************************************************************************
* In this section, we estimate the Figure 3: Kurtosis of log earning
use "PS4_Clean_Moments.dta", clear

* Regression
reghdfe kurtosis_log_income i.age_center i.year, noconstant
parmest, saving("$outdir/Figure_3_Coefficients.dta", replace)

use "$outdir/Figure_3_Coefficients.dta", clear
* Keep only age_center coefficients
keep if strpos(parm, "age_center") > 0

* Extract the age value from the parm string
gen age = real(regexs(1)) if regexm(parm, "([0-9]+)\.age_center")
* Scale all estimates so that the estimate at age 40 is equal to 4.3407
gen scale_factor = .
quietly summarize estimate if age == 40
local scale = r(mean) 
replace scale_factor = estimate - `scale' + 4.3407

* Figure 3: Kurtosis of log earning
graph twoway (line scale_factor age, sort lcolor(black) lwidth(thick)), ///
ytitle("", size(small)) ///
xtitle("Age") title("Figure 3: Kurtosis of log earning") ///
xlabel(25(5)60) xscale(range(25 60))

graph export "$outdir/Figure_3.png", width(2000) replace
export delimited "$outdir/Figure_3.csv",replace









