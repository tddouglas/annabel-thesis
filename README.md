## Analysis
We've implemented this using [SkLearn's Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) function. 

We've chosen the `liblinear` solver as it uses a "[One-vs-Rest](https://refactored.ai/microcourse/notebook?path=content%2F06-Classification_models_in_Machine_Learning%2F02-Multivariate_Logistic_Regression%2Fmulticlass_logistic-regression.ipynb#1.-One-Vs-Rest-(OVR))" scheme evaluating each independent variable individually. Given the importance a single material exports can have to a nation, it is best to choose a solver utilizing this OVR scheme.

We've chosen low `C` values and an "l2" `penalty` as the dataset is inherently small (limited to # of countries). These choices attempt to address overfitting at a slight penalty to test accuracy. 


## Annabel's Description
Attached is my dataset. Only the first two tabs really matter. I have controls in the final few tabs but I figured for now I’ll leave them be, considering I can’t even figure out how to run a test with just independent variables.
 
The only positive progress I made was when I isolated independent variables (in the case of the two graphs below, first Manganese Ore and then Copper Ore). When I tried to pull in multiple independent variables the tests would stop working. Ultimately I have two questions: “What is the probability of a country receiving a swap if they export at least 5% of any critical green input globally?” and, “What is the probability of a country receiving a swap given how much of a critical green export they buy from China?”. The two first ‘master’ tabs have all the data, if I could only figure out how to run it..

![](static/image001.png)
![](static/image002.png)

Tyler Notes:
- Removed commas from country names to allow CSV file to work