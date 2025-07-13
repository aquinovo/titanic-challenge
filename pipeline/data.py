import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    cols_to_drop = ['Name', 'Ticket', 'Cabin']
    df = df.drop(cols_to_drop, axis=1)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    return df

def add_one_hot_features(df):
    embark_dummies = pd.get_dummies(df['Embarked'])
    sex_dummies = pd.get_dummies(df['Sex'])
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Class")
    df = df.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
    df = df.join([embark_dummies, sex_dummies, pclass_dummies])
    return df
