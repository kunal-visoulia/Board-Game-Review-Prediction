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

## [Linear Regression](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)
When the target variable that we’re trying to predict is **continuous**(Real valued like predict selling price of a house), we call the learning problem a regression problem.

The other most common type of supervised learning problem is called the **classification problem** where we predict **discrete-valued outputs** (Example, if a tumor is malignant(1) or benign(0). So that's a zero-one valued discrete output.). 

Linear regression is a **Supervised Learning Algorithm**(we're given the "right answer" for each of our training examples) used for finding linear relationship between _target_(predicted house price, denoted by **y**) and one or more _predictors_(attributes from the training data set, like size of house in feets, no. of bedrooms, etc denoted by **'x'** or by **Feature Vector X** ).


![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/2.png)


**'h'** represents the **Hypothesis Function**

There are two types of linear regression- Simple and Multiple.

### Simple Linear Regression or Linear Regression with One Variable
Simple linear regression is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x _horizontal axis_) and dependent(y _vertical axis_) variable. The red line in the above graph is referred to as the best fit straight line. Based on the given data points, we try to plot a line that models the points the best. 

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/3.png)

The line can be modelled based on the linear equation shown below.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/9.png)

#### [Cost Function for Simple Linear Regression](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd)
We minimize the cost function(J) to find the best values for our hypothesis function parameters.
What we actually do is we minimize the error between the predicted value and the actual value using MSE function.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/5.png)

The **Mean Squared Error(MSE) Function**. **'m'** represents the size of training dataset. 

Now, using this MSE function we are going to change the values of theta_0 and theta_1 such that the MSE value settles at the minima.

#### Gradient Descent for Simple Linear Regression
For Parameter Learning ,i.e., updating theta_0 and theta_1 to reduce the cost function(MSE), we use **Gradient Descent**.

The idea is that we start with some values for theta_0 and theta_1 and then we change these values iteratively to reduce the cost. Gradient descent helps us on how to change the values.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/7.png)
![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/8.png)

**_alpha_ represents the learning rate(decides on how fast the algorithm converges to the minima.)**

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/4.png)

To draw an analogy how gradient descent works, imagine a pit in the shape of U and you are standing at the topmost point in the pit and your objective is to reach the bottom of the pit. **You can only take a discrete number of steps to reach the bottom**

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/6.png)

Sometimes the cost function can be a non-convex function where you could settle at a local minima but for linear regression, it is always a convex function, and thus settle always at global minima.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/10.png)

### Multiple/Multivariate Linear Regression 
The multivariable form of the hypothesis function accommodating these multiple features is as follows:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/12.png)

Here, in case of Boardgame features, theta_0 can represent average_rating, theta_1 can represent average_weight, theta_2 can represent playing_time, theta_3 can represent min_playing_time, and so on, while x1,x2,x3.... represent respective values for the features in the training dataset.
**x0 is not part of dataset and assumed to be 1**

To understand better, let's take an example from Andrew Ng's Machine Learning Course for predicting Housing Prices.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/15.png)

m = the number of training examples in the dataset

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/13.png" width="440px" />

#### [Cost Function and Gradient Descent for MVLR](https://www.ritchieng.com/multi-variable-linear-regression/)

<img src="https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/14.png" width="440px" />

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/16.png)

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
















