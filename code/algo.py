from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class algo():
    
    
    """
    This class will contain functions that will preprocess the data-set
    and fit machine learning models.
   
    """
    
    def education_feature(data):
        """
        cleaning eduction freature as it contains many categories.
        grouping all the basics
        """
        data['education'] = np.where(data['education'].str.contains('basic'), 'Basic', data['education'])
    
  
    def create_dummy_variables(data):
        """
        create dummy variables for categorical features
        """
        
        cat_vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
        
        for var in cat_vars:
            cat_list='var'+'_'+var
            cat_list = pd.get_dummies(data[var], prefix=var)
            data1=data.join(cat_list)
            data=data1
            
        ##keeping the right columns
        data_vars=data.columns.values.tolist()
        to_keep=[i for i in data_vars if i not in cat_vars]
        
        data_final=data[to_keep]
        
        return data_final
    
    def standardiser(data):
        """
        This function will split the dataframe into X and y and
        also standardise the features (important for linear regression
        and make emsemble maching learning methods quicker to train)
        Note, make sure your predictor is labelled as 'y'.
        """
        
        X = data.drop(['y'], axis = 1)
        y = data.loc[:, 'y']
        
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        X_std = pd.DataFrame(X_std, columns = X.columns)
        
        return y, X_std
    
    def preprocessing(X_std, y):
        """
        splitting data into training and testing (80:20)
        inputs require X_std and y 
        """
        validation_size = 0.20 ##80:20 for testing
        seed = 7
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_std, y, test_size=validation_size, 
                                 random_state=seed)
        
        return X_train, X_test, y_train, y_test
    
    def SMOTE(X_train, y_train):
        """
        Applying over-sampling technique on the training data only.
        Therefore no data will be lost in the testing dataset
        """     
        os = SMOTE(random_state=0)
        columns = X_train.columns
        os_data_X,os_data_y=os.fit_sample(X_train, y_train)
        os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
        os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
        
        return os_data_X, os_data_y
        
   
    def model_fit(os_data_X, os_data_y, X_test, y_test):
        """
        This function will fit the model using the training data 
        which has been oversampled. 
        Note, this is using no hyperparamter tunning.
        Returns the y_pred, y_pred_proba, accuracy, confusion matrix
        and classification report
        """
        logreg = LogisticRegression(solver = 'lbfgs')
        logreg.fit(os_data_X, os_data_y.values.ravel())
        y_pred = logreg.predict(X_test)
        y_pred_proba = logreg.predict_proba(X_test)
        accuracy = ('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return y_pred, y_pred_proba, accuracy, conf_matrix, class_report
    
    
    def run_algo(data):
        """
        This function will run the entire algorithm including
        pre-processing the dataframe for it to be machine learning
        ready.
        Input: data
        Outputs will be y_pred, y_pred_proba, accuracy, conf_matrix
        and classification_report
        """
        algo.education_feature(data)
        data_final = algo.create_dummy_variables(data)
        y, X_std = algo.standardiser(data_final)
        X_train, X_test, y_train, y_test = algo.preprocessing(X_std, y)
        os_data_X, os_data_y = algo.SMOTE(X_train, y_train)
        y_pred, y_pred_proba, accuracy, conf_matrix, class_report = algo.model_fit(os_data_X, os_data_y, X_test, y_test)
        
        return y_pred, y_pred_proba, accuracy, conf_matrix, class_report 
        
        
        
    


