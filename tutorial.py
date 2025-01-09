import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# downalod kaggle dataset
def download_data():
    api = KaggleApi()
    api.authenticate()

    api.competition_download_files('titanic','data_titanic')

    with zipfile.ZipFile('data_titanic/titanic.zip', 'r') as zip_ref:
        zip_ref.extractall('data_titanic/')


# first challenge tutorial
def tutorial():

    pd.set_option('display.max_columns', None) # set full view of columns
    train_data = pd.read_csv('data_titanic/train.csv')
    test_data = pd.read_csv('data_titanic/test.csv')

    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)

    print("% of women who survived:", rate_women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)

    print("% of men who survived:", rate_men)

    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission_tutorial.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    if  not os.path.exists('data_titanic/titanic.zip'):
        download_data()
        print('Data downloaded')
    else:
        print('Data already downloaded')

    tutorial()
