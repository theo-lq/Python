import numpy as np
import pandas as pd





def partial_dependance_plot_points(model, X, column, class_of_interest=1):
    """Return the abscissa and ordinate of the partial dependance plot of the variable column in the X dataframe or array, for a classifier model.
    
    Usage::
        >>> #we assume we have fitted a model (e.g. a RandomForestClassifier) for a classification problem
        >>> x, y = partial_dependance_plot_points(model, X_test, "Age")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(x, y)
    :param: model: a classifier model e.g. from sklearn`
            X : A pandas DataFrame or numpy array
            column : A string
            class_of_interest : An integer
    :rtype: two list of floats
    """
    
    
    def change_value(value):
        """Replace in X the values of the column column by value"""
        X_new = X.copy()
        X_new[column] = value * np.ones(X_new.shape[0])
        return X_new
    
    abscissa = np.linspace(start=min(X[column]), stop=max(X[column]), num=100)
    ordinate = []

    for value in abscissa:
        X_new = change_value(value)
        y_pred_proba = model.predict_proba(X_new)
        ordinate.append(np.mean(y_pred_proba[:, class_of_interest]))
    
    return abscissa, ordinate