import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

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
    
    
    def stats_summary(data):
        """
        Applying summary statistics on data 
        """
        stats = data.describe()
        
        return stats
    
    def class_dist(data):
        """
        This function shows how the classes are distributed across the data-set
        """
        group_data = data.groupby('class').size()
        return group_data
    
    def univariate_plt(data, plot_type):
        """
        Function will plot a chart for individual variable.
        plot_type variable will contain either 'hist' or 'box',
        for example etl.data_analysis.univariate(data, 'hist') for
        histogram.
        """
        
        data.plot(kind = plot_type, subplots=True, layout=(2,2),
                     sharex=False, sharey= False)
        plt.show()
     
    def multivariate_plt(data):
        """
        Plots the interaction between the different variables using
        scatter matrix from pandas plotting module.
        """
        scatter_matrix(data, alpha=0.7)
        plt.show()
    
    
   
    