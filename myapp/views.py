from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def irisModel():
    global val_predictions, val_y, iris_model
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
def accuracyTesting():
    truth_state_series = pd.Series(val_predictions == val_y)
    total_test_count = len(val_y)
    hit_count = len(truth_state_series[truth_state_series == True])
    miss_count = len(truth_state_series[truth_state_series == False])
    return (hit_count/total_test_count)*100


#### Using Your Model
def testSpecies(sepalLength = 0, sepalWidth = 0, petalLength = 0, petalWidth = 0):
    features = np.array([[sepalLength , sepalWidth , petalLength , petalWidth ]])
    val_predictions = iris_model.predict(features)
    return val_predictions[0]




def home(request):
    context = {'res': "Please enter above values"}
    values = []
    if request.GET:
        values.append(request.GET.get('slength',''))
        values.append(request.GET.get('swidth',''))
        values.append(request.GET.get('plength',''))
        values.append(request.GET.get('pwidth',''))
        try:
            if len(values[0]) == 0 or len(values[1]) == 0 or len(values[2]) == 0 or len(values[3]) == 0:
                context = {'res': 'Please Fill All Values'}
                return render(request, 'home.html', context)
            elif eval(values[0]) == 0 or eval(values[1]) == 0 or eval(values[2]) == 0 or eval(values[3]) == 0:
                context = {'res': 'Please Enter Non-zero value'}
                return render(request, 'home.html', context)
            else:
                irisModel()
                predicted_species = testSpecies(sepalLength = eval(values[0]), sepalWidth = eval(values[1]),  petalLength = eval(values[2]), petalWidth = eval(values[3]))
                context = {'res' : predicted_species, 'sl': values[0], 'sw':values[1] , 'pl':values[2] , 'pw':values[3],'text':'The Predicted Species is : ', 'ok':True}
        except:
            context = {'res': "Enter Valid Real Number"}
            return render(request, 'home.html', context)

    return render(request, 'home.html', context)
   


