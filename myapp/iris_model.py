#### Imports
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



#### File reading
url ="https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/iris.csv"
col_name_list = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
Iris_data =  pd.read_csv(url, header = None, names = col_name_list)





#### DataSplitting
Iris_data_df_to_array = Iris_data.to_numpy()
# Feature Matrix
X = Iris_data_df_to_array[:, :(len(Iris_data_df_to_array[0,:])-1)]
# Target Matrix
y = Iris_data_df_to_array[:,4]



#### Model Validation
# Data Splitting between training set and test set(Validation Data)
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0)


#### Model Building and Prediction
# model definition 
iris_model = LogisticRegression()
# model fitting
iris_model.fit(train_X,train_y)
# model prediction
val_predictions = iris_model.predict(val_X)
# Creating Prediction DataFrame Table
predicted_df = pd.DataFrame(val_X ,columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
predicted_df["Predicted Species"] = val_predictions.reshape(len(val_predictions),1)
predicted_df["Actual Species"] = val_y
predicted_df['True State'] = (val_y == val_predictions).reshape(len(val_predictions),1)







#### Accuracy Testing
truth_state_series = pd.Series(val_predictions == val_y)
total_test_count = len(val_y)
hit_count = len(truth_state_series[truth_state_series == True])
miss_count = len(truth_state_series[truth_state_series == False])
def accuracy(hit=0,total_test=0):
    return (hit/total_test)*100
print('--',hit_count, '-- prediction out of --', total_test_count,\
    '-- was HIT and --', miss_count,'-- MISSED with hit rate of --',\
    accuracy(hit=hit_count,total_test=total_test_count),' %--', sep = '')


    
#### Using Your Model
def testSpecies(sepalLength = 0, sepalWidth = 0, petalLength = 0, petalWidth = 0):
    features = np.array([[sepalLength , sepalWidth , petalLength , petalWidth ]])
    val_predictions = iris_model.predict(features)
    return val_predictions[0]
