set logtype text
set more off
*******************************************************************************
* Econ 810: Spring 2025 Advanced Macroeconomics 
* by J. Carter Braxton 
* Author: Zachary Orlando and Cutberto Frias Sarraf
*******************************************************************************

*******************************************************************************

* Set working directory (change to your directory)
cd "\\sscwin\dfsroot\labusers\friassarraf\Documents\STATA\810_ECON_Carter\Homework Two"
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
* i11110 	= Individual Labor Earnings

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
local oldnames x11101LL x11102 i11110 d11102LL d11101 d11112LL d11106 d11107 x11104LL d11105 d11109 e11104 e11101
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
	
	import delimited "CPIAUCSL.csv"
	
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
keep if inrange(year,1978,1997) // Year must be between 1978 and 1997 
keep if oversample_id == 11		// Drop the SEO sample 
// keep if relationship_head == 1 	// Consider household heads only 
keep if inrange(age, 25,55)		// Consider only observations where the head was working-age

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

* Declare the panel structure
tsset person_id year

* A full-time job in the U.S. requires 40 hours per week. Over the course of a full year (52 weeks) we obtain that the amount of working hours should be 2,080. Now if we consider vacations and holidays we could set a lower bound of 1,850 hours.

* Create a new variable full_time and initialize it with missing values
gen full_time = .
* Set full_time = 1 if the person is working (working_status == 1) and working hours are higher than 1,850
replace full_time = 1 if workings_hours > 1850 & working_status == 1
// bysort working_status, summarize workings_hours

* Create a new variable to store the household income difference, initializing with missing values
gen hh_income_diff = .
* Sort data by person ID and year to ensure correct time ordering
bysort person_id (year): replace hh_income_diff = hh_income - hh_income[_n-1] ///
    if full_time == 1 & full_time[_n-1] == 1 

* Calculate the average change in earnings and standard deviation for individuals with valid hh_income_diff
summarize hh_income_diff if full_time == 1 & !missing(hh_income_diff)

* Store the summary results in scalars for hh_income_diff
scalar avg_change = r(mean)
scalar sd_change = r(sd)

* Calculate the mean and standard deviation of hh_income
summarize hh_income if full_time == 1 & !missing(hh_income)

* Store the summary results in scalars for hh_income
scalar avg_income = r(mean)
scalar sd_income = r(sd)

* Display the calculated results for hh_income_diff and hh_income
display "Average Change in Labor Earnings: " avg_change
display "Standard Deviation of Change in Labor Earnings: " sd_change
display "Average Labor Earnings: " avg_income
display "Standard Deviation of Labor Earnings: " sd_income

* Create a temporary dataset to store the result for export
clear
set obs 1

* Create variables to store the results and assign values
gen variable_name = "Average Change in Labor Earnings"
gen value = avg_change

* Create variables for the standard deviation of change
gen sd_variable_name = "Standard Deviation of Change in Labor Earnings"
gen sd_value = sd_change

* Create variables for the mean and standard deviation of hh_income
gen income_variable_name = "Average Labor Earnings"
gen income_value = avg_income

gen income_sd_variable_name = "Standard Deviation of Labor Earnings"
gen income_sd_value = sd_income

* Export the result to a CSV file
export delimited using "$outdir/avg_change.csv", replace



