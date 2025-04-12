set logtype text
set more off
*******************************************************************************
* Econ 810: Spring 2025 Advanced Macroeconomics 
* by J. Carter Braxton 
* Author: Zachary Orlando and Cutberto Frias Sarraf
*******************************************************************************
* Set cutoff for working full-time 
*A full-time job in the U.S. requires 40 hours per week. 
* Over the course of a full year (52 weeks) we obtain that the amount of working hours should be 2,080. 
* Now if we consider vacations and holidays we set a lower bound of 1,850 hours.
local full_time_hours = 1850 
*******************************************************************************
* Set working directory (change to your directory)
cd "C:\Users\zacha\Documents\2025 Spring\Advanced Macroeconomics\J. Carter Braxton\Homework\Homework Three\"
pwd   // Check the current working directory
dir   // List files in the directory

local current_dir : pwd
* Set input location 
global indir "`current_dir'"
* Set output location 
global outdir "$indir/images"
*******************************************************************************
* Variables Used: 
* Key: 
* x11101LL 	= Individual identifier 
* x11102 	= Household Identification Number
* year 		= year of survey

* Income: 
* i11110 	= Individual Labor Earnings

* Working Hours
* e11101 = Annual Work Hours of Individual

* Demographics:
* e11104 	= Working status (0 = working, 1 = not)
* d11101 	= Age 
* d11106 	= Number of Persons in Household 
* d11107 	= Number of Children in Household

* Additional vars used for sample restrictions: 
* x11104LL 	= Oversample identifier (11 = Main Sample, 12 = SEO)
*******************************************************************************

cd "$indir"

* Load data 
use "pequiv_long.dta",clear

********************************************************************************
* Clean PSID for analysis
********************************************************************************
* Rename variables
local oldnames x11101LL x11102 i11110 e11101 e11104 d11101 d11106 d11107 x11104LL
local newnames person_id hh_id hh_income working_hours working_status age hh_size hh_children oversample_id 

local i = 1
foreach oldvar in `oldnames' {
    local newvar: word `i' of `newnames'
    rename `oldvar' `newvar'
    local ++i
}

keep person_id year hh_id hh_income working_hours working_status age hh_size hh_children oversample_id 

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
keep if inrange(year,1978,1997) // Year must be between 1978 and 1997 
keep if oversample_id == 11		// Drop the SEO sample 
keep if inrange(age, 25,54)		// Consider only observations where the head was working-age (30 years)

* Drop if demographics are missing
gen missing_dem = 0
foreach var in `newnames' {
	
	replace missing_dem = 1 if missing(`var')
}

drop if missing_dem == 1 
drop missing_dem

drop hh_id age price_level oversample_id 
********************************************************************************
* Create panel of Job-stayers and Job-Losers for TWFE event study design 
********************************************************************************
* Declare the panel structure
tsset person_id year

* Create a full-time work indicator representing whether a person was working and whether they worked full-time
gen full_time = (working_hours >= `full_time_hours' & working_status == 1)

* Drop individuals who skip years in the PSID 

gen year_skip = (year != year[_n-1] + 1 & person_id == person_id[_n-1])
bysort person_id (year): egen skipped_years = max(year_skip)

*drop if skipped_year // taken out

* Job Losers *******************************************************************

* Define job loss as working hours < 75% of last year's working hours, having worked full time for the past three years  
bysort person_id (year): gen job_loss = (working_hours <= 0.75 * working_hours[_n-1] & full_time == 0 & full_time[_n-1] == 1 & full_time[_n-2] == 1 & full_time[_n-3] == 1)

* Record the year of first job loss 
bysort person_id (year):  gen job_loss_count = sum(job_loss)
gen temp = year if job_loss_count == 1 & job_loss == 1
bysort person_id (year): egen year_of_job_loss = max(temp) 

drop temp

* Generate job loser indicator
bysort person_id (year): egen job_loser = max(job_loss)

* Job Stayers ******************************************************************

* Define a job stayer as an individual who has never experienced a job loss in the PSID 
* as well as: 
* they have an employment spell for at least four consecutive years in the sample. 

bysort person_id (year): gen employment_indicator = (full_time == 1 & full_time[_n-1] == 1 & full_time[_n-2] == 1 & full_time[_n-3] == 1)

bysort person_id (year): egen employed_four_years = max(employment_indicator) 

gen job_stayer = (job_loser == 0 & employed_four_years == 1)

* Record the year of the first >= four-year employment spell began.
bysort person_id (year): gen employment_indicator_count = sum(employment_indicator)
bysort person_id (year): gen temp = year if employment_indicator_count == 1 & employment_indicator_count[_n-1] == 0
bysort person_id (year): egen year_of_employment = max(temp) 

drop if job_loser == 0 & job_stayer == 0 

drop temp
********************************************************************************
* Drop if HH Size Grows
* It seems to me that shocks can either increase HH size (children) or 
* decrease it (death).
********************************************************************************
* Record if the person's household size grew 
gen hh_size_break = 0 
replace hh_size_break =  1 if (hh_size > hh_size[_n-1] & person_id == person_id[_n-1]) 

bysort person_id: egen hh_size_changed = max(hh_size_break) 

drop if hh_size_changed

* Count number of remaining individuals
bysort person_id (year): gen n =_n

tabstat person_id if n == 1,by(job_stayer) statistics(N)

drop n
******************************************************************************** 
* Run the regression: 
******************************************************************************** 
* For job losers, this is centred on the year of job loss
gen event_time = year - year_of_job_loss if job_loser == 1

* For job-stayers, this is centred on the fourth year of employment
replace event_time = year - (year_of_employment) if job_stayer == 1

* In davis and Von Wachter, the dependent variable is loss relative to earnings four years prior. 
* Here, I normalize by earnings three years prior to job loss. 

gen temp = hh_income if event_time == -3

bysort person_id (year): egen initial_income = max(temp)

gen normalized_income = hh_income/initial_income

* Probably need to winsorize normalized income because of the possibility of extreme values 
winsor2 normalized_income, cuts(1 99)

drop temp 
* Generate the treatment group 
gen treated = (job_loser == 1)

* Generate interaction terms capturing the differential trend between job-losers and job-stayers
* four years pre job-loss to 10 years post job-loss
forvalues i = -3/10 {
	local int_label = cond(`i' < 0, "m" + string(abs(`i')), "p" + string(`i'))
    gen treat_time_`int_label' = treated * (event_time == `i')
}

keep if inrange(event_time,-3,10)

reghdfe normalized_income_w treat_time_m2-treat_time_m1 treat_time_p0-treat_time_p10,a(person_id year)

********************************************************************************
* Plot our event study
********************************************************************************
// Extract coefficient estimates and standard errors
matrix b_est = e(b)
matrix V_est = e(V)

// Create a new coefficient matrix with the reference period included
matrix b_plot = J(1, 14, .)
// Fill pre-treatment periods
matrix b_plot[1,1] = 0  // -3 (reference)
matrix b_plot[1,2] = b_est[1,1]  // -2
matrix b_plot[1,3] = b_est[1,2]  // -1
// Fill post-treatment periods
forvalues i = 0/10 {
    matrix b_plot[1,`i'+4] = b_est[1,`i'+3]  // 0 to 10
}

// Create a matrix of standard errors
matrix se_plot = J(1, 14, .)
// Fill pre-treatment standard errors
matrix se_plot[1,1] = 0  // reference period
matrix se_plot[1,2] = sqrt(V_est[1,1])
matrix se_plot[1,3] = sqrt(V_est[2,2]) 
// Fill post-treatment standard errors
forvalues i = 0/10 {
    matrix se_plot[1,`i'+4] = sqrt(V_est[`i'+3,`i'+3])
}

// Name the coefficients for proper x-axis labels
matrix colnames b_plot = -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10

// Create the plot
coefplot matrix(b_plot), se(se_plot) ///
    vertical ///
    ciopts(recast(rcap) lcolor(black%70)) ///
    mcolor(navy%70) msymbol(circle) ///
    yline(0, lpattern(dash) lcolor(red)) ///
    xline(4, lpattern(dash) lcolor(gray)) ///
    xtitle("Years after layoff") ///
    connect(l) lcolor(black) lwidth(medthick) /// Added connecting line
    ytitle("Earnings Loss" "Relative to Initial Earnings") ///
    title("Effect of Layoff on Earnings") ///
    note("Note: 95% confidence intervals shown" "Normalized Earnings Winsorized at the 1% level") ///
    legend(off)

// Save the graph
cd "$outdir"
graph export "event_study_plot.png", replace width(2000) height(1500)	
