# DataInsight
==============

About the repository
--------------

**Postdoctoral Researcher at Deutsches Elektronen-Synchrotron**

*Creators*:
Christian Contreras-Campana, PhD <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

Dedicated area for the development data insight in the area of predictive analytics from churn prediction, fraud detection, and general maching learning modeling

This repository contains an example jupyter notebook. The DataInsight repository include the following topics:

- Data science visualization
- Model optimization
- Feature selection
- Machine learning model explanation and interpretation
- Deep learning
- Computer vision
- Reinforcement learning
- Topic modeling
- Sentiment analysis
- Time series analysis
- Cluster analysis
- A/B testing with Bayesian inference
- Survivial analysis
- Utilities toolkits


Installation
--------------

The DataInsight toolkit repository may be install using pip.
Use the following command:
```
pip install -e git+https://github.com/chrisjcc/DataInsight#egg=DataInsight --process-dependency-links
```

Development
--------------

For the purposes of code development the pip install version will
need to be removed. Use the following command:
```
pip uninstall DataInsight
```
Then a local copy of the git repository will need to be checked out.
These are the instructions on how to install and run local package:

1. Inside repository:

```
python setup.py install
```
2. Start Jupyter notebook or restart Jupyter notebook depending on your
circumstance.

3. Import the package using:
```
from neural_network import *

# Existing folder

cd existing_folder

git init

git remote add origin git+https://github.com/chrisjcc/DataInsight

git add .

git commit -m "Initial commit"

git push -u origin master

# Existing Git repository

cd existing_repo

git remote rename origin old-origin

git remote add origin git+https://github.com/chrisjcc/DataInsight

git push -u origin --all

git push -u origin --tags


Available hyper-parameter optimization techniques:

- GridSearchCV/RandomizedSearchCV: Exhaustive/Randomized parameter searches
- HyperOpt: Sequetial model-based optimization via Bayesian optimization with Gaussian Process or Tree-sturctured of Parzen Estimators (TPE) 
- Bayes_Opt Bayesian optimization constructing a posterior distribution of functions for gaussian process, 
- SkOpt: Bayesian optimization with acquisition function: Expected improvement (EI), Lower Confidence bound (LCB), and Probability Improvement (PI) 
- Optunity: Primarily based on PSO
