# data_utils.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # Load dataset
    data = pd.read_csv('penguins.csv')

    # Encode OriginLocation
    labelencoder = LabelEncoder()
    data['OriginLocation'] = labelencoder.fit_transform(data['OriginLocation'])

    # Columns to impute
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']

    # Handle missing values
    mean_imputer = SimpleImputer(strategy='mean')
    median_imputer = SimpleImputer(strategy='median')

    mean_imputed = mean_imputer.fit_transform(data[mean_cols])
    median_imputed = median_imputer.fit_transform(data[median_col])

    # Convert to DataFrame
    mean_imputed_df = pd.DataFrame(mean_imputed, columns=mean_cols)
    median_imputed_df = pd.DataFrame(median_imputed, columns=median_col)

    # Replace original columns with imputed values
    imputed_data = pd.concat([mean_imputed_df, median_imputed_df], axis=1)
    data[['CulmenDepth', 'CulmenLength', 'BodyMass', 'FlipperLength']] = imputed_data

    return data
