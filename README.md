# COMPGW02 Web Economics - Real-Time-Bidding (RTB) on the iPinYou dataset (Group 21)
This repository contains the source code, group and individual reports for the Group Project of "COMPGW02 Web Economics" course for UCL's MSc in Business Analytics (academic year 2017-2018).


## Group Members
* Achilleas Sfakianakis
* Akis Thomas
* Dwane van der Sluis

## Description
In this project, we worked on an online advertising problem. We helped advertisers to form a bidding strategy in order to place their ads online in a real-
time bidding (RTB) system. RTB mechanism to buy and sell ads is the most significant evolution in recent years in display and mobile advertising. RTB essentially facilitates 
buying an individual ad impression in real time, automatically triggered by a user's visit. Although other types of auctions, such as the first price auction, are also popular, RTB exchanges typically employ the second price auction model.
For this project we used the iPinYou dataset. This dataset was released by the iPinYou Information Technologies Co., Ltd (iPinYou), which was founded in 2008 and is currently 
the largest DSP in China. The dataset includes logs of ad auctions, bids, impressions, clicks, and final conversions.

## Results
Our best performing model was an ensemble model of an XGBoost model and a "Multi - Random Forest" model combined with a linear bidding strategy. The "Multi - Random Forest" model
accounted for the very large size of the training set (~2.4M) and the extreme class imbalance (only 1793 clicks, CTR ~0.07%) by training multiple Random Forest classifiers on
different subsets of the training set (each subset contained all clicks plus 1800 non-clicks). Model performed a noteworthy performance on the validation set buying 165 clicks (out of
202 total clicks) with an impressive CTR of 0.22%. Performance was also evaluated in a "Multi-Agent" real-time environment where the 31 group teams of the course where competing 
against each other to win auctions on the test set. Until the last day of the final submission, our team managed to rank on the __1st__ position, getting 41 clicks.

## Repository Structure

Most of the code is inside jupiter notebooks. 

* common - utility code for splitting / loading or joining datafiles 
* Q1_Data_Exploration_individual
* Q2_Basic_Bidding_Strategy
* Q3_Linear_bidding_strategy
* Q4_best_strategies_individual
* Q5_further_developed_bidding_strategy
* old_code - holds old no longer used code
* Bids placed - holds records of what we have submitted  
* tmp - directory for images etc to be written to

## Other

### Code Location
github: https://github.com/DwanevanderSluis/COMPGW02_Web_Economics.git

### Temp Data Files sharing location  
https://1drv.ms/f/s!AvZgRVgV-b7thpkNJ_yEeEOjX6B0lQ

### Write up is in overleaf 
https://www.overleaf.com/13750448qkcgchjbgjqg#/53221961/

### Data Used for Project (prepared by lecturer)
https://liveuclac-my.sharepoint.com/personal/ucabyw4_ucl_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fucabyw4%5Fucl%5Fac%5Fuk%2FDocuments%2Fwe%5Fdata%2Ezip&parent=%2Fpersonal%2Fucabyw4%5Fucl%5Fac%5Fuk%2FDocuments&slrid=04044c9e%2De09d%2D5000%2Dc6da%2Da231b90d989a

### original complete iPinYou dataset 
http://data.computational-advertising.org/



