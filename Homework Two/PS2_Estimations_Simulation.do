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

cd "$indir"

* Load data 
import delimited "Dataframe_Simulations.csv", clear

xtset individual year

encode group, gen(group_numeric)

xtreg earnings ib2.group_numeric##ib5.year, fe robust

* Export the result to a LaTeX file
// estout using "results_est.tex", replace tex

