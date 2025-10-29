import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data():
    data = pd.read_csv('penguins.csv')
    labelencoder = LabelEncoder()
    data['OriginLocation'] = labelencoder.fit_transform(data['OriginLocation'])
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']
    mean_imputer = SimpleImputer(strategy='mean')
    median_imputer = SimpleImputer(strategy='median')
    mean_imputed = mean_imputer.fit_transform(data[mean_cols])
    median_imputed = median_imputer.fit_transform(data[median_col])
    mean_imputed_df = pd.DataFrame(mean_imputed, columns=mean_cols)
    median_imputed_df = pd.DataFrame(median_imputed, columns=median_col)
    imputed_data = pd.concat([mean_imputed_df, median_imputed_df], axis=1)
    data[['CulmenDepth', 'CulmenLength', 'BodyMass', 'FlipperLength']] = imputed_data
    scaler = MinMaxScaler()

    data[['CulmenDepth', 'CulmenLength', 'BodyMass', 'FlipperLength']] = scaler.fit_transform(
        data[['CulmenDepth', 'CulmenLength', 'BodyMass', 'FlipperLength']]
    )
    return data
