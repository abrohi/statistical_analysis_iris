from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
    
    
    def model_fit(X_train_std, Y_train):
        """
        This function will fit the model using the training data 
        for X and Y. Note, this is using no hyperparamter tunning.
        """
        logreg = LogisticRegression(multi_class= "multinomial", 
                                    solver = "lbfgs")
        logreg.fit(X_train_std, Y_train)
        
        return logreg
    
    def model_accuracy(model, X_validation_std, Y_validation):
        """
        Calculate the model accuracy (i.e. the number of times
        the model made the correct classification using the testing
        data and model. 
        Note, this is using the score() function in sklearn.
        """
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_validation_std, Y_validation)))
        
    def confusion_matrix(model, X_validation_std, Y_validation):
        """
        Calcualate a confusion matrix to see the accuracy by group.
        Input required is model, testing data.
        """
        Y_pred = model.predict(X_validation_std)
        
        confusion_matrix = confusion_matrix(Y_validation, Y_pred)
        return confusion_matrix

    


