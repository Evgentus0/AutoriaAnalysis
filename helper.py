import pandas as pd


def get_means(data: pd.DataFrame, column_name):
    values = data[column_name]
    count = values.count()
    min = values.min()
    max = values.max()
    mean = values.mean()
    sd = values.std()
    quantile1 = values.quantile(q=.25)
    quantile2 = values.quantile(q=.5)
    quantile3 = values.quantile(q=.75)

    return count, min, max, mean, sd, quantile1, quantile2, quantile3



