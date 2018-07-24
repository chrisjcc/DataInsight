## Introduction Bike-Sharing data challenge

Bike-Share is a bike share service provider where users can take and return bikes at any of the 100 stations on their network. The company wants to leverage their data to better understand and, hopefully, optimize their operations. They are interested in exploring the potential of machine learning to predict the number of bikes taken and returned to each station.

The company would like to use the output of the predictive model to help the logistics team schedule the redistribution of bikes between stations. This in turn will help ensure there are bikes and return docks available when and where users need them.

To arrange this schedule, they need an estimation of the net change in the stock of bikes at a station for the time window between two pick-up/drop-off visits so the station may face the upcoming demand. Naturally, the number of visits will depending on the intensity of the use of that station, the use of other stations in the network, as well as resources available.

As a first step towards tackling this challenge, we are to develop a model capable of outputing the net rate of bike renting for a given station (i.e. net rate is defined as trips ended minus trips started at the station for a given hour). That is, at any time the logistic department of the company should be able to make a statement such as "In the next hour, the net stock of bikes at a station A will change by X."

Results will be accompanied by the root-mean-square error (RMSE) metric as a measure of the performance of each model. But we should keep in mind that RMSE has zero tolerance for outlier data points which don't belong. RMSE relies on all data being right and all are counted as equal. That means one stray point that's way out in left field is going to totally ruin the whole calculation.
