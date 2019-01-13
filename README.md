# Board-Game-Review-Prediction
## Using Linear Regression Model And A Random Forest Regressor Machine Learning Algorithms
In this project, I built a sytem that reviews/rates boardgames. 

When I analyzed the dataset, for correaltions between various attributes using correlation matrix and plotting a heatmap, I found not much strong correlations.

![]()

# DATASET
In the dataset we had around 81,000 different boardgames.

**_DATA PREPROCESSING_**

I was able to remove those games that were never rated because they were never published.
While training the either model, I removed attributes like name, id, bayes_average_rating and average_rating(because that's what we wanted to predict) as they were not providing any useful information. 
https://github.com/ThaWeatherman/scrapers/tree/master/boardgamegeek/games.csv


