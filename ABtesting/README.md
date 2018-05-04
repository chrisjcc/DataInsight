# Udacity A/B Testing Experiment

 Fully defined design, analyze the results, and propose a high-level follow-on experiment.


## Experiment Design

### Metric Choice

List which metrics used as invariant metrics and evaluation metrics.

For each metric, explain both why you did or did not use it as an invariant metric and why you did or did not use it as an evaluation metric. Also, state what results you will look for in your evaluation metrics in order to launch the experiment.


** Choosing invariant metrics:

- Number of cookies: There is no difference between the control and experimental group when trying to view the course overview page, therefore the number of cookies should be the same in each group. 

- Number of clicks: There is no difference the control and experimental group beforey they click the "Start free trial' button, therefore  the number of cookies should be the same in each group.

- click-through-probability: Since there is no difference in the 'number of cookies' and 'number of clicks' metrics between the control and experimental group, there will be no difference in this metric. 

** Metrics not choosen:

- Number of user-ids: The experimental group is expect to have fewer users enrolled in the free-trial, thefore consistency is not expected between the groups. 

- Cross conversion: The experimental group is expected to have fewer user-ids enrolled in the free trial, thefore consistency is not expected between the groups. 

- Retention: The experimental group is expected to have more user-ids remain past the free-trial period and have fewer enrolled in the free-trial, therefore consistency is not expected between the groups. 

- Net conversion: The experimental group is expected to have relatively more user-ids remain enrolled past the free-trial period, therefore consistency is not expected between the groups. 

** Choosing Evaluation Metrics:

- Cross conversion: Fewer user-ids are expected to complete the checkout and enrollend in the free-trial in the experimental group, but the number of unique cookies to click "Start free trial" button should be the same. For this reason, the gross conversion ratio should be higher for the control group than for the experimental group. To lauch the experiment, we need the experiment value to be at least 0.001 lower than the control value. 

- Retention: At least as many user-ids in the experimental group are expected to reamin enrolled past the free-trial period, and fewer are expected to complete the free-trial checkout. For this reason, the retention ratio is expected to be higher for the experimental group than the control group. To launch the experiment, we need the experiment value to be no more than 0.001 less than the control value. 

- Net conversion: At  least as many user-ids in the experimental group are expected to reamin	 enrolled past the free-trial period, and the number of unique cookies to click "Start free trial" button should be the same. For this reason the net conversion ratio should not decrease for the experiment. To lanuch the experiment, we need the experiment value to be no more than 0.0075 lower than the control value. 

* Metrics not chosen

- Number of cookies: There is no difference between the control and experimental group when trying to view the course overview page, and for this reason the number of cookies should be the same in each group. 

- Number of user-ids: Although it is a valid evaluation metric, I am not choosing it because it is not normalized and if we have slighly different sized experimental and control groups, its accuracy will decrease. Therefore, it is not a useful metric for comparison as a chosen evaluation metric.

- Number of clicks: There is no difference between the control and experimental group before they click the "Start free trial" button, therefore the number of clicks should be the same in each group. 

- Click-through-probability: Since there is no difference in the 'Number of cookies' and 'number of clicks' metrics between the experimental and control group, there will be no difference in this metric.

### Measuring Standard Deviation 

List the standard deviation of each of the evaluation metrics.

For each of your evaluation metrics, indicate whether you think the analytic estimate would be comparable to the the empirical variability, or whether you expect them to be different (in which case it might be worth doing an empirical estimate if there is time). Briefly give your reasoning in each case.


### Sizing 

Number of Samples vs. Power
Indicate whether you will use the Bonferroni correction during your analysis phase, and give the number of pageviews you will need to power you experiment appropriately.

Duration vs. Exposure

Indicate what fraction of traffic you would divert to this experiment and, given this, how many days you would need to run the experiment.

Give your reasoning for the fraction you chose to divert. How risky do you think this experiment would be for Udacity?


## Experiment Analysis

### Sanity Checks 

For each of your invariant metrics, give the 95% confidence interval for the value you expect to observe, the actual observed value, and whether the metric passes your sanity check. 

For any sanity check that did not pass, explain your best guess as to what went wrong based on the day-by-day data. 

## Result Analysis

### Effect Size Tests 

For each of your evaluation metrics, give a 95% confidence interval around the difference between the experiment and control groups. Indicate whether each metric is statistically and practically significant.

### Sign Tests 

For each of your evaluation metrics, do a sign test using the day-by-day data, and report the p-value of the sign test and whether the result is statistically significant. 

* Summary

State whether you used the Bonferroni correction, and explain why or why not. If there are any discrepancies between the effect size hypothesis tests and the sign tests, describe the discrepancy and why you think it arose.

* Recommendation 

Make a recommendation and briefly describe your reasoning.

## Follow-Up Experiment 

Give a high-level description of the follow up experiment you would run, what your hypothesis would be, what metrics you would want to measure, what your unit of diversion would be, and your reasoning for these choices.


