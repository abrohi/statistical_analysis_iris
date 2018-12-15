from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class algo():
    
    
    """
    This class will contain functions that will preprocess the data-set
    and fit machine learning models.
   
    """
    
    def preprocessing(data):
       
        """
        This function will convert the data-set into a numpy array and 
        then split the data into train and validation, and finally 
        standardise the features. One will have arrays 
        X_train_std, X_validation_std, Y_train and Y_validation
        
        """
        
        ##spliting data by train and test
        ##create a validation dataset
        # Split-out validation dataset
        array = data.values ##convert dataset to numpy array
        X = array[:,0:4] ##get the first 4 columns for x values
        Y = array[:,4] ##get last column for y variable
        validation_size = 0.20 ##80:20 for testing
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, 
                                         random_state=seed)
        ##standardising the features - 
        ##seeing how many stdev is point away from mean
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_validation_std = sc.fit_transform(X_validation)
        
        return X_train_std, X_validation_std, Y_train, Y_validation
        

    


