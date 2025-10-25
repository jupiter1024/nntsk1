import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def confusion_matrix_manual(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    cm = np.array([[TP, FN],
                   [FP, TN]])
    return cm
def signum(net):
  if net>0:
    return 1
  elif net<=0:
    return -1

def Preprocessing_After_input2(data,Species1,Species2, feature1,feature2,lr,epoch,mse_threshold,is_bias):
    
    s1=data[data['Species']==Species1]
    s2=data[data['Species']==Species2]
    new_data=pd.concat([s1,s2])

    final_data = new_data[['Species',feature1, feature2]]
    np.random.seed(42)
    y=final_data['Species']
    X=final_data.drop('Species',axis=1)

    labelencoder1 = LabelEncoder()
    y_encoded = labelencoder1.fit_transform(y)
    y_encoded = np.where(y_encoded == 0, -1, 1)
    y_encoded = pd.DataFrame(y_encoded, columns=['Species'])

    X_train, X_test, y_train, y_test = train_test_split( X, y_encoded,test_size=0.4,random_state=42,stratify=y_encoded ,shuffle=True)

    if is_bias:
      X_train["bias"]=1
      X_test["bias"] = 1
    else:
      X_test["bias"]=0
      X_train["bias"]=0

    X_train_np=X_train.to_numpy()
    X_test_np=X_test.to_numpy()
    y_train_np=y_train.to_numpy()
    y_test_np=y_test.to_numpy()

    w2=np.random.rand(3,1)
    
    for _ in range(epoch):
        y_pred_all = []
        for i, x in enumerate(X_train_np):
            x = x.reshape(-1, 1)
            y_pred = np.dot(x.T, w2)
            y_pred_all.append(y_pred)

            e = y_train_np[i] - y_pred
            w2 = w2 + lr * e * x  

        y_pred_all = np.array(y_pred_all).flatten()
        error = mse(y_train_np, y_pred_all)
        print(f"MSE: {error}")

        if error <= mse_threshold:
            print(f"✅ MSE threshold reached: {error:.6f} <= {mse_threshold}")
            break

    y_pred_test2 = []

    for i, x in enumerate(X_test_np):
        x = x.reshape(-1, 1)
        net = np.dot(x.T, w2)
        y_pred_test2.append(net)

    y_pred_test2 = np.array(y_pred_test2).flatten()
    y_pred_test_labels = np.where(y_pred_test2 >= 0, 1, -1)
    accuracy = np.mean(y_pred_test_labels.flatten() == y_test_np.flatten()) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    cm = confusion_matrix_manual(y_test_np.flatten(), y_pred_test_labels.flatten())

    return accuracy, final_data, w2, X_test_np, y_test_np, cm

