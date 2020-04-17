import numpy as np
import pandas as pd





def One_Hot_Significative(df, column, alpha=0.05):
    """Perform One-Hot-Encoding in the dataframe df for the column column, but only for the most represented modalities.

    Usage::
        >>> list_of_modalities = ["red", "blue", "green"]
        >>> variable = np.random.choice(list_of_modalities, size=100, p=[0.01, 0.59, 0.4])
        >>> df = pd.DataFrame({"variable": variable})
        >>> df = One_Hot_Significative(df, "variable")

    :param df: A pandas DataFrame.
    :param column: A string.
    :param alpha: A float.
    :rtype: A pandas DataFrame.
    """

    threshold = round(df.shape[0] * alpha)

    possibilities = df[column].value_counts()

    for i in range(len(possibilities)):
        if possibilities[i] >= threshold:
            candidate = possibilities.index[i]
            name = column + '_' + candidate
            df[name] = df[column].apply(lambda x : 1 if x == candidate else 0)

    answer = df.drop(column, axis=1)
    return answer





def One_Hot_All(df, min_modalities=5, alpha=0.05):
    """Perform One-Hot-Encoding in the dataframe df for all the categorial variable. Full One-Hot-Encoding if the number of modalities is less than the min_modalities number, else the significative One-Hot-Encoding.

    Usage::
        >>> list_of_modalities = ["red", "blue", "green", "yellow", "orange", "black"]
        >>> variable_3 = np.random.choice(list_of_modalities[:3], size=1000, p=[0.01, 0.59, 0.4])
        >>> variable_6 = np.random.choice(list_of_modalities, size=1000, p=[0.01, 0.29, 0.4, 0.1, 0.18, 0.02])
        >>> numeric = np.random.randn(1000)
        >>> df = pd.DataFrame({"numeric": numeric, "variable_3": variable_3, "variable_6": variable_6})
        >>> df = One_Hot_All(df)


    :param df: A pandas DataFrame.
    :param min_modalities: An integer.
    :param alpha: A float.
    :rtype: A pandas DataFrame.
    """

    columns = df.columns

    for column in columns:
        if df.dtypes[column] == object:
            value = df[column].value_counts()
            if len(value) > min_modalities:
                df = One_Hot_Significative(df, column, alpha)
            else:
                df = pd.get_dummies(df, columns=[column])

    return df
