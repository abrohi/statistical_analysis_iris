import pandas as pd

class data_analysis():
    """
    The set of functions within the class will load data-set, run statistical summary analysis 
    and plot visualisations to help one understand the data-set better.
    """
    
    

    def data_read():
        """
        This function will allow one to read in the iris dataset as a dataframe
        """

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

        dataset = pd.read_csv(url, names= names, index_col=None)

        return dataset