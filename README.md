Titanic disaster kaggle Challenge

Challenge link: https://www.kaggle.com/competitions/titanic

Here you can find the official tutorial code for the challenge (tutorial.py) and my solution (main.py).

First of all I download the zip file and extract it in a folder.
Then, I do an EDA step to understand the data and the features.

The data set is composed by the following columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

Amongst them there are some columns that have null values, precisely: Age, Cabin, Fare and Embarked. 
Therefore, I perform a data cleaning step to fill the missing values.
I used the median by class for the Age column, the word 'Unknown' for the Cabin column, the median for the Fare column and the mode for the Embarked column.

After that, I perform a feature engineering step to create new features that could be useful for the model.
For example, a column that indicates the title of the passenger, one for Fare Per Person, one for Family Size and one if the passanger is alone or not.
Title could be useful because it often encodes social status, age and gender information.
Fare Per Person to better capture individual ticket cost.
Family Size and Alone or not to capture some possible patter like parents that save their children first.

Now, I drop some column that I think are not useful for the model, like PassengerId, Name, Ticket and Cabin.
I decide to keep Embarked because it could be useful to capture some information about the social status of the passenger.

Then, I perform a feature encoding step to convert categorical features into numerical ones. 
After that, I scale the features to reduce the range of values.

Now, through a feature selection step I keep only the most important features for the model.

Looking at the graph I can see that a dummy evaluator which return always true if the passenger is a woman would reach an accuracy of 75%, therefore my goal is to overcome this score.

I define an ensemble model composed by a Random Forest, LightGBM and XGBoost with Logistic Regression as final estimator.
I choose these models because:
- RandomForest for robust, easy-to-interpret results and handling various feature types without scaling.
- LightGBM for fast, high-performance modeling, especially with large datasets and categorical features.
- XGBoost for excellent predictive performance with boosting, regularization, and feature importance.
- Logistic Regression as a baseline for linear relationships and to test if more complex models are needed.

My solution reach an accuracy of 83.84%.



