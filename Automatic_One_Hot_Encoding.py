import numpy as np
import pandas as pd

def One_Hot_Significative(df, column, alpha=0.05):
    threshold = round(df.shape[0] * alpha)
    
    possibilities = df[column].value_counts()
    
    for i in range(len(possibilities)):
        if possibilities[i] >= threshold:
            candidate = possibilities.index[i]
            name = column + '_' + candidate
            df[name] = df[column].apply(lambda x : 1 if x == candidate else 0)
    
    answer = df.drop(column, axis=1)
    return answer


def One_Hot_All(df, alpha=0.05):
    columns = df.columns
    
    for column in columns:
        if df.dtypes[column] == object:
            value = df[column].value_counts()
            if len(value) > 5:
                df = One_Hot_Significative(df, column, alpha)
            else:
                df = pd.get_dummies(df, columns=[column])
    
    return df