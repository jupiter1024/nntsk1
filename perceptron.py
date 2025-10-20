import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def Preprocessing_After_input_perceptron(data,Species1,Species2, feature1,feature2,lr,epoch,mse_threshold,is_bias):



    s1=data[data['Species']==Species1]
    s2=data[data['Species']==Species2]

    new_data=pd.concat([s1,s2])


    final_data = new_data[['Species',feature1, feature2]]


    np.random.seed(42)

    y=final_data['Species']
    X=final_data.drop('Species',axis=1)







    labelencoder1 = LabelEncoder()
    y_encoded = labelencoder1.fit_transform(y)

# Convert 0 → -1 and 1 → 1
    y_encoded = np.where(y_encoded == 0, -1, 1)


    y_encoded = pd.DataFrame(y_encoded, columns=['Species'])



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.4,           # 20 out of 50 → 40% test
        random_state=42,
        stratify=y_encoded ,shuffle=True      # keeps class proportions equal
    )


    unique, train_counts = np.unique(y_train, return_counts=True)
    unique, test_counts = np.unique(y_test, return_counts=True)

    test_counts







    is_bias=False

    w=np.random.randn(3,1)






    if is_bias:
     X_train["bias"]=1
     X_test["bias"] = 1
    else:
     X_test["bias"]=0
     X_train["bias"]=0

    X_train.to_numpy()

    X_train_np=X_train.to_numpy()
    X_test_np=X_test.to_numpy()
    y_train_np=y_train.to_numpy()
    y_test_np=y_test.to_numpy()

    for _ in range(epoch):
        for i, x in enumerate(X_train_np):
            x = x.reshape(-1, 1)          # column vector
            net = np.dot(x.T, w)          # weighted sum
            y_pred = signum(net)
            e = y_train_np[i] - y_pred    # correct way
            if e != 0:
                w = w + lr * e * x



    for _ in range(epoch):
        for i, x in enumerate(X_train_np):
            x = x.reshape(-1, 1)          # column vector
            net = np.dot(x.T, w)          # weighted sum
            y_pred = signum(net)
            e = y_train_np[i] - y_pred    # correct way
            if e != 0:
                w = w + lr * e * x
    y_pred_test = []
    for i, x in enumerate(X_test_np):
        x = x.reshape(-1, 1)
        net = np.dot(x.T, w)
        y_pred = signum(net)
        y_pred_test.append(y_pred)

# Convert to flat NumPy array
    y_pred_test = np.array(y_pred_test).flatten()



    accuracy = np.mean(y_pred_test.flatten() == y_test.to_numpy().flatten()) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


    unique, counts = np.unique(y_pred_test, return_counts=True)
    print(dict(zip(unique, counts)))
    return accuracy,final_data,w




def signum(net):
  if net>0:
    return 1
  # elif net==0:
  #   return 0
  elif net<0:
    return -1


