# Board-Game-Review-Prediction
## Using Linear Regression Model And A Random Forest Regressor Machine Learning Algorithms
In this project, I built a sytem that reviews/rates boardgames. 

## RESULT
When I analyzed the dataset, for correaltions between various attributes using correlation matrix and plotting a heatmap, I found not much strong correlations.

![](https://github.com/kunal-visoulia/Board-Game-Review-Prediction/blob/master/images/1.png)

I found out that on the given dataset the **Linear Regression Model** had a ***Mean Sqaured Error*** of over 2 while,

**Random Forest Regressor Model** had a ***Mean Sqaured Error*** of around 1.4.

When I tested model on random game data from the dataset, On some cases Linear Regression worked better(it predicted a closer rating to actual user rating than predicted by random forest regressor) but ***on most cases Random Forest Regressor Model worked better obviously due to its lower Mean Sqaured Error***

## DATASET
In the dataset we had around 81,000 different boardgames.

URL: https://github.com/ThaWeatherman/scrapers/tree/master/boardgamegeek/games.csv

**_DATA PREPROCESSING_**

I was able to remove those games that were never rated because they were never published.
While training the either model, I removed attributes like name, id, bayes_average_rating and average_rating(because that's what we wanted to predict) as they were not providing any useful information. 

## Linear Regression


