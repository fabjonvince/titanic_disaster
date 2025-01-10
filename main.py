import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier


# downalod kaggle dataset
def download_data():
    api = KaggleApi()
    api.authenticate()

    api.competition_download_files('titanic','data_titanic')

    with zipfile.ZipFile('data_titanic/titanic.zip', 'r') as zip_ref:
        zip_ref.extractall('data_titanic/')


def main():

    # exploratory data analysis
    pd.set_option('display.max_columns', None) # set full view of columns
    train_data = pd.read_csv('data_titanic/train.csv')
    print(train_data.head())

    test_data = pd.read_csv('data_titanic/test.csv')
    print(test_data.head())

    data = pd.concat([train_data, test_data]) # concat to perform preprocessing and better visualization
    print(data.info())
    print(data.describe())

    sns.barplot(x='Pclass', y='Survived', data=train_data)
    plt.title('Survival rate by class')
    plt.show()

    sns.barplot(x='Sex', y='Survived', data=train_data)
    plt.title('Survival rate by Sex')
    plt.show()

    # Preprocessing
    # handle missing age value by filling with median age of each class
    data['Age'] = data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    # Bin age into categories
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 65, 99], labels=['Child', 'Teenager', 'Adult', 'Elderly'])

    # handle missing embarked with mode
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # extract Deck from Cabin
    data['Deck'] = data['Cabin'].str[0].fillna('Unknown')

    # handle missing fare with median
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # create new feature 'Title' to encode social status, age and gender
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) # extract title from name
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare') # replace rare titles
    data['Title'] = data['Title'].replace('Mlle', 'Miss') # replace french titles
    data['Title'] = data['Title'].replace('Ms', 'Miss') # replace Ms with Miss
    data['Title'] = data['Title'].replace('Mme', 'Mrs') # replace french titles

    # create new feature 'FamilySize' to encode family size of each passenger on board
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # calculate fare per person
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']

    # create new feature 'IsAlone' to encode if passenger is alone or not
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # drop unnecessary columns
    data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # encode categorical features
    cat_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'Deck']
    data = pd.get_dummies(data, columns=cat_features, drop_first=True)


    # split data back to train and test
    train = data[:len(train_data)] # data that were originally in train_data are kept in train
    test = data[len(train_data):] # data that were originally in test_data are kept in test
    X = train.drop(['Survived'], axis=1) # drop target column
    Y = train['Survived']
    X_test = test.drop(['Survived'], axis=1) # drop target column

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # feature selection
    rf = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=10)
    rfe.fit(X, Y)
    X=X[X.columns[rfe.support_]]
    X_test = X_test[X.columns]

    # split train in train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # model training
    # stack RandomForest, LightGBM and XGBoost with Logistic Regression as final estimator
    stack_model = StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)),
        ('lgb', lgb.LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.01, num_leaves=10, verbose=-1)),
        ('xgb', XGBClassifier(random_state=42))
    ], final_estimator=LogisticRegression())
    stack_model.fit(X_train, y_train)
    stack_pred = stack_model.predict(X_val)

    # evaluation
    print('Stacking accuracy:', accuracy_score(y_val, stack_pred))
    print('Stacking classification report:', classification_report(y_val, stack_pred))

    # cross-validation
    cv_scores = cross_val_score(stack_model, X, Y, scoring='accuracy')
    print('Cross-validation scores:', cv_scores.mean())

    # predict test data
    test_data['Survived'] = stack_model.predict(X_test)

    # save submission
    output = test_data[['PassengerId', 'Survived']]
    output.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")





if __name__ == '__main__':
    # download kaggle dataset
    if  not os.path.exists('data_titanic/titanic.zip'):
        download_data()
        print('Data downloaded')
    else:
        print('Data already downloaded')

    main()
