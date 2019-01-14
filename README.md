# Board-Game-Review-Prediction
## Using Linear Regression Model And A Random Forest Regressor Machine Learning Algorithms
In this project, I built a sytem that reviews/rates boardgames. 

## RESULT
When I analyzed the dataset, for correaltions between various attributes using correlation matrix and plotting a heatmap, I found not much strong correlations.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/1.png)

I found out that on the given dataset the **Linear Regression Model(Multivariate)** had a ***Mean Sqaured Error*** of over 2 while,

**Random Forest Regressor Model** had a ***Mean Sqaured Error*** of around 1.4.

When I tested model on random game data from the dataset, On some cases Linear Regression worked better(it predicted a closer rating to actual user rating than predicted by random forest regressor) but ***on most cases Random Forest Regressor Model worked better obviously due to its lower Mean Sqaured Error***

## DATASET
In the dataset we had around 81,000 different boardgames.

URL: https://github.com/ThaWeatherman/scrapers/tree/master/boardgamegeek/games.csv

**_DATA PREPROCESSING_**

I was able to remove those games that were never rated because they were never published.
While training the either model, I removed attributes like name, id, bayes_average_rating and average_rating(because that's what we wanted to predict) as they were not providing any useful information. 

The mathematics and theory for Linear Regression is quite long, so here are the hyperlinks for the page:
### 1. [LINEAR REGRESSION](#linear-regression) 
- [Simple Linear Regression or Linear Regression with One Variable](#simple-linear-regression-or-linear-regression-with-one-variable)
- [Cost Function for Simple Linear Regression](#cost-function-for-simple-linear-regression)
- [Gradient Descent for Simple Linear Regression](#gradient-descent-for-simple-linear-regression)
- [Multivariate Linear Regression](#multivariate-linear-regression)
- [Cost Function and Gradient Descent for MVLR](#cost-function-and-gradient-descent-for-mvlr)
- [Gradient Descent in Practice](#gradient-descent-in-practice)
- [Polynomial Regression](#polynomial-regression)
- [Normal Equation for Learning Parameters](#normal-equation-for-learning-parameters)

### 2. [RANDOM FORREST REGRESSION](#random-forrest-regression)
- [Decision Tree Regression](#decision-tree-regression)
- [Bootstrap method and Bootstrap Aggregation](#bootstrap-method-and-bootstrap-aggregation)
- [From Bagging to Random Forest](#from-bagging-to-random-forest)


## [Linear Regression](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)
When the target variable that we’re trying to predict is **continuous**(Real valued like predict selling price of a house), we call the learning problem a regression problem.

The other most common type of supervised learning problem is called the **classification problem** where we predict **discrete-valued outputs** (Example, if a tumor is malignant(1) or benign(0). So that's a zero-one valued discrete output.). 

Linear regression is a **Supervised Learning Algorithm**(we're given the "right answer" for each of our training examples) used for finding linear relationship between _target_(predicted house price, denoted by **y**) and one or more _predictors_(attributes from the training data set, like size of house in feets, no. of bedrooms, etc denoted by **'x'** or by **Feature Vector X** ).


![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/2.png)


**'h'** represents the **Hypothesis Function**

There are two types of linear regression- Simple and Multiple.

### Simple Linear Regression or Linear Regression with One Variable
Simple linear regression is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x _horizontal axis_) and dependent(y _vertical axis_) variable. The red line in the above graph is referred to as the best fit straight line. Based on the given data points, we try to plot a line that models the points the best. 

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/3.png" width="440px"/>

The line can be modelled based on the linear equation shown below.

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/9.png" width="240px"/>

#### [Cost Function for Simple Linear Regression](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd)
We minimize the cost function(J) to find the best values for our hypothesis function parameters.
What we actually do is we minimize the error between the predicted value and the actual value using MSE function.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/5.png)

The **Mean Squared Error(MSE) Function**. MSE can never be negative.
**'m'** represents the size of training dataset. 

Now, using this MSE function we are going to change the values of theta_0 and theta_1 such that the MSE value settles at the minima.

#### Gradient Descent for Simple Linear Regression
For Parameter Learning ,i.e., updating theta_0 and theta_1 to reduce the cost function(MSE), we use **Gradient Descent(the other way is through [Normal Equation](https://www.ritchieng.com/multi-variable-linear-regression/))**.

The idea is that we start with some values for theta_0 and theta_1 and then we change these values iteratively to reduce the cost. Gradient descent helps us on how to change the values.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/7.png)
![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/8.png)

**_alpha_ represents the learning rate(decides on how fast the algorithm converges to the minima.)**

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/4.png" width="440px"/>

To draw an analogy how gradient descent works, imagine a pit in the shape of U and you are standing at the topmost point in the pit and your objective is to reach the bottom of the pit. **You can only take a discrete number of steps to reach the bottom**

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/6.png)

Sometimes the cost function can be a non-convex function where you could settle at a local minima but for linear regression, it is always a convex function, and thus settle always at global minima.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/10.png)

### Multivariate Linear Regression 
The multivariable form of the hypothesis function accommodating these multiple features is as follows:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/12.png)

Here, in case of Boardgame features, theta_0 can represent average_rating, theta_1 can represent average_weight, theta_2 can represent playing_time, theta_3 can represent min_playing_time, and so on, while x1,x2,x3.... represent respective values for the features in the training dataset.
**x0 is not part of dataset and assumed to be 1**

To understand better, let's take an example from Andrew Ng's Machine Learning Course for predicting Housing Prices.

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/15.png" width="500px"/>

m = the number of training examples in the dataset

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/13.png" width="440px" />

#### [Cost Function and Gradient Descent for MVLR](https://www.ritchieng.com/multi-variable-linear-regression/)

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/14.png" width="440px" />

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/16.png" width="440px"/>

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/17.png)

## Gradient Descent in Practice 
![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/18.png)

Two techniques to help with this are feature scaling and mean normalization
### I - Feature Scaling (MVLR) and Mean Normalization
***Feature scaling*** involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. 

***Mean normalization*** involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/19.png)

### II - The Learning Rate
If α is too small: slow convergence.

If α is too large: ￼may not decrease on every iteration and thus may not converge.

### Polynomial Regression
![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/20.png)

### Normal Equation for Learning Parameters
The normal equation formula is given below:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/21.png)

The following is a comparison of gradient descent and the normal equation:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/22.png)

**Normal Equation Noninvertibility**

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/23.png)

## [Random Forrest Regression](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd)
Random Forest is a **supervised learning algorithm**.Unlike linear models, random forests are able to capture non-linear interaction between the features and the target.
The random forest model is a type of additive model that makes predictions by combining decisions from a sequence of base models. 

More formally we can write this class of models as:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/24.png)
![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/25.png)

Here, each base classifier is a *simple decision tree*.

>It is a type of ensemble(of decision trees) machine learning algorithm called **[“bagging” (Bootsrap Aggregation)](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)**.

>**Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.**

### [Decision Tree Regression](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)
The **decision tree** is a simple machine learning model for getting started with regression tasks.

It uses a tree-like model of decisions, **_BUT, How can an algorithm be represented as a tree?
_**
For this let’s consider a very basic example that uses titanic data set for predicting whether a passenger will survive or not. Below model uses 3 features/attributes/columns from the data set, namely sex, age and sibsp (number of spouses or children along).This methodology is more commonly known as **learning decision tree from data**. 

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/26.png)

>each *internal* (non-leaf) node denotes a test on an attribute, each *branch* represents the outcome of a test, and each *leaf* (or terminal) node holds a class label. The topmost node in a tree is the root node. 
Above tree is called **Classification tree** as the target is to classify passenger as survived or died. **Regression trees*** are represented in the same manner, just they predict continuous values like price of a house.

*Although, a real dataset will have a lot more features and this will just be a branch in a much bigger tree, but you can’t ignore the simplicity of this algorithm.*

>In general, Decision Tree algorithms are referred to as **CART** or Classification and Regression Trees.

**Advantages of CART**

- Simple to understand, interpret, visualize.
- Decision trees implicitly perform variable screening or feature selection.
- Can handle both numerical and categorical data(take one of a limited, and usually fixed, number of possible values.).
- Nonlinear relationships between parameters do not affect tree performance.

**Disadvantages of CART**

- Decision-tree learners can create over-complex trees that do not generalize the data well and thus lead to**overfitting**.
- Decision trees **can be unstable** because small variations in the data might result in a completely different tree being generated. This is called variance, which needs to be _lowered by methods like **bagging** and boosting_.
- Decision tree learners create biased trees if some classes dominate. It is therefore _recommended to balance the data set prior_ to fitting with the decision tree.
- They are greedy. They choose which variable to split on using a greedy algorithm that minimizes error.

### [Bootstrap method and Bootstrap Aggregation](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)
The bootstrap is a powerful statistical method for estimating a quantity from a data sample. This is easiest to understand if the quantity is a descriptive statistic such as a mean or a standard deviation.

Bootstrap Aggregation, is a simple and very powerful ensemble method which can be, Just like the decision trees themselves,  used for classification and regression problems.

An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model.

Bootstrap Aggregation is used to reduce the variance for those algorithm that have high variance like classification and regression trees (CART).

**But why do decision trees show such high variance?**
>If the number of levels is too high i.e a complicated decision tree, the model tends to overfit.

Intuitively, it can be understood in this way.

>When there are too many decision nodes to go through before arriving at the result i.e number of nodes to traverse before reaching the leaf nodes is high, the conditions that you are checking against becomes multiplicative. That is, the computation becomes (condition 1)&&(condition 2)&&(condition 3)&&(condition 4)&&(condition5).
Only if all the conditions are satisfied, a decision is reached. As you can see, this will work very well for the training set as you are continuously narrowing down on the data. The tree becomes highly tuned to the data present in the training set.
But when a new data point is fed, even if one of the parameters deviates slightly, the condition will not be met and it will take the wrong branch.

When bagging with decision trees, we are less concerned about individual trees overfitting the training data. For this reason and for efficiency, the individual decision trees are grown deep (e.g. few training samples at each leaf-node of the tree) and the trees are not pruned(size of tree is not reduced to overcome overfitting). These trees will have both high variance and low bias. These are **important characterize of sub-models when combining predictions using bagging**.

The only parameters when bagging decision trees is the number of samples,i.e,**number of trees to include**. This can be chosen by increasing the number of trees on run after run until the accuracy begins to stop showing improvement. Very large numbers of models may take a long time to prepare, but will not overfit the training data.

**How Bagging of the CART algorithm works:**
Let’s assume we have a sample dataset of 1000 instances.

1. Create many (e.g. 100) random sub-samples of our dataset with replacement.
2. Train a CART model on each sample.
3. Given a new dataset, calculate the average prediction from each model.

For example, if we had 5 bagged decision trees that made the following class predictions for a in input sample: blue, blue, red, blue and red, we would take the most frequent class and predict blue.

## [From Bagging to Random Forest](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)
Random Forests are an improvement over bagged decision trees.

A problem with decision trees like CART is that they are greedy. They choose which variable to split on using a greedy algorithm that minimizes error. As such, even with Bagging, the decision trees can have a lot of structural similarities and in turn have high correlation in their predictions.

Combining predictions from multiple models in ensembles works better if the predictions from the sub-models are uncorrelated or at best weakly correlated.

To solve tree correlation we allow random forest to randomly choose only m predictors in performing the split. Now the bagged trees all have different randomly selected features to perform cuts on. Therefore, the feature space is split on different predictors, decorrelating all the trees.











