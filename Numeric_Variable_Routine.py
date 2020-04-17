import numpy as np
import pandas as pd





def replace_outliers_fill_na(df, column, outliers=True, outliers_alpha=0.05, fill_na=True):
    """Fill na value in a dataframe df at the column column if asked, and manage outliers by clipping the value between the minimum and maximum outliers_alpha percentile if asked.

    Usage::
        >>> vector = np.random.randn(1000)
        >>> index = np.random.choice(range(1000), size=10, replace=False)
        >>> vector[index] = 5 * vector[index] * np.random.randn(10)
        >>> df = pd.DataFrame({"numeric": vector})
        >>> df = replace_outliers_fill_na(df, "numeric")

    :param df: A pandas DataFrame.
    :param column: A string.
    :param outliers: A boolean.
    :param outliers_alpha: A float, positive.
    :param fill_na: A boolean.
    :rtype: A pandas DataFrame.
    """

    new_df = df.copy()
    serie = new_df[column].dropna()

    if outliers:
        max_percentile = np.percentile(serie, 100-outliers_alpha)
        min_percentile = np.percentile(serie, outliers_alpha)
        new_df[column] = np.clip(new_df[column], min_percentile, max_percentile)

    if fill_na:
        new_df[column].fillna(serie.median())

    return new_df





def check_numeric_variable(df, outliers=True, outliers_alpha=0.05, fill_na=True):
    """Fill na value in a dataframe df for all the columns if asked, and manage outliers by clipping the value between the minimum and maximum outliers_alpha percentile if asked.

    Usage::
        >>> vector = np.random.randn(1000)
        >>> index = np.random.choice(range(1000), size=10, replace=False)
        >>> vector[index] = 5 * vector[index] * np.random.randn(10)
        >>> list_of_modalities = ["red", "blue", "green"]
        >>> modalities = np.random.choice(list_of_modalities, size=1000)
        >>> df = pd.DataFrame({"numeric": vector, "string": modalities})
        >>> df = check_numeric_variable(df)

    :param df: A pandas DataFrame.
    :param outliers: A boolean.
    :param outliers_alpha: A float, positive.
    :param fill_na: A boolean.
    :rtype: A pandas DataFrame.
    """

    new_df = df.copy()
    columns = new_df.columns

    for column in columns:
        if new_df.dtypes[column] == float64:
            new_df = replace_outliers_fill_na(new_df, column, outliers, outliers_alpha, fill_na)

    return new_df
