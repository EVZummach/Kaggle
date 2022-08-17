import pandas as pd

def preprocess(df, train=True):
    print(f'Input data length: {df.shape[0]}')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    onehot = pd.get_dummies(df[['Sex', 'Embarked']])
    input_columns = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] if train else ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    df_out = pd.merge(df[input_columns], onehot, on='PassengerId').dropna()
    print(f'Output data length: {df.shape[0]}')
    df_out = feature_engineering(df_out)
    print(f'Output Columns: {df_out.columns}')
    return df_out

def feature_engineering(df):
    df['Family Size'] = df['SibSp']+df['Parch']
    df['Fare/Age'] = df['Fare']/df['Age']
    df['SibSp/Parch'] = df['SibSp']/(df['Parch']+1)
    return df