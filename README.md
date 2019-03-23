# cebd1160_project_template

| Name | Date |
|:-------|:---------------|
|Michael D'Itri | 2019-03-21|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: `Final_script.py`
- Results figure/saved file: `figures/`
- Dockerfile for your experiment: `Dockerfile`
- runtime-instructions in a file named `RUNME.md`

-----

## Research Question

Based on 13 different chemical compenents of wine, can we predict from which of the 3 different cultivators the wine originates? 

### Abstract

The data is comprised of thirteen different measurements taken for different constituents found in the three types of wine (3 different origins). Leveraging this data, we may be able to determine the relationship between the chemical composition of wine, which could then be used to classify wine that has lost its label. Based on the performance of the Gaussian Naive Bayes model, we found that the model was fairly consistent and could be used as a relatively good model for determining the origin of the wine.

### Introduction

The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators.
There are thirteen different measurements taken for different constituents found in the three types of wine.
(https://scikit-learn.org/stable/datasets/index.html#wine-dataset)

### Methods

The method for modelling the data was Gaussian Naive Bayes built into scikit-learn.
Pseudocode: https://scikit-learn.org/stable/modules/naive_bayes.html

The reason why we chose Naive Bayes is because it works well with a small training set, is extremely fast.
Also, The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
(https://scikit-learn.org/stable/modules/naive_bayes.html)

Secondly, we found another group who ran the same data set through multiple classifiers and showed that Naive Bayes had one of the best performances (>95%). (https://jonathonbechtel.com/blog/2018/02/06/wines/)

### Results

![alt text](https://github.com/mikeditri/class9/blob/master/figures/Gaussian_NB_with_Accuracy.png)

We can see in the confusion matrix that this model has a accuracy well over 95% and the number of misclassified wines in very low thus illustrating that this model is a good choice for classifying the wines.

### Discussion

The method used seems to easily predict wine origin based on its chemical composition when using all features in the analysis. With a accuracy rate over 95%, it seems to be more than adequate to be able to identify which vinyard a non-labeled wine comes from. There is always ways to improve and just because the accuracy rate is high it doesn't mean that it will perform this well with new data. In order to make this better, I would try to to run this model on new data, measure the performance and then if the models accuracy is deteriorating, I would tune the selected features to improve the model. 

### References

The links referenced were included in my discussion, above.

-------
