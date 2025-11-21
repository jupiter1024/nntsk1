import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data():
    data = pd.read_csv(r'penguins.csv')
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
    return data, labelencoder, mean_imputer, median_imputer, scaler

def prepare_data_for_nn():
    d, origin_location_encoder, mean_imputer, median_imputer, scaler = load_and_preprocess_data()
    label_encode = LabelEncoder()
    d['Species'] = label_encode.fit_transform(d['Species'])    
    Fs = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    x, y = d[Fs].values, d['Species'].values
    return x, y, label_encode, Fs, origin_location_encoder, mean_imputer, median_imputer, scaler

def split_data_by_class(x, y, train_samples=30, test_samples=20):
    unique_classes = np.unique(y)
    x_test_parts, y_test_parts, x_train_parts, y_train_parts = [], [], [], []
    for class_label in unique_classes:
        class_msk = (y == class_label)
        class_y = y[class_msk]
        class_x = x[class_msk]
        x_train_parts.append(class_x[:train_samples])
        x_test_parts.append(class_x[train_samples:train_samples + test_samples])
        y_train_parts.append(class_y[:train_samples])
        y_test_parts.append(class_y[train_samples:train_samples + test_samples])
    x_train, y_train, x_test, y_test = np.vstack(x_train_parts), np.hstack(y_train_parts), np.vstack(x_test_parts), np.hstack(y_test_parts)
    print(f"training set: {x_train.shape[0]} samples")
    print(f"test set: {x_test.shape[0]} samples")
    print(f"Classes in training: {np.unique(y_train)}")
    return x_train, y_train, x_test, y_test

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]

def preprocess_sample(sample_dict, features, origin_location_encoder, mean_imputer, median_imputer, scaler):
    # Create a DataFrame with the sample
    sample_df = pd.DataFrame([sample_dict])
    
    # Encode OriginLocation
    if 'OriginLocation' in sample_df.columns:
        sample_df['OriginLocation'] = origin_location_encoder.transform(sample_df['OriginLocation'])
    
    # Reorder columns to match features order
    sample_df = sample_df[features]
    
    # Separate columns for imputation
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']
    
    # Apply imputation (in case of missing values)
    mean_imputed = mean_imputer.transform(sample_df[mean_cols])
    median_imputed = median_imputer.transform(sample_df[median_col])
    
    # Combine imputed data
    mean_imputed_df = pd.DataFrame(mean_imputed, columns=mean_cols)
    median_imputed_df = pd.DataFrame(median_imputed, columns=median_col)
    imputed_data = pd.concat([mean_imputed_df, median_imputed_df], axis=1)
    
    # Add OriginLocation back
    imputed_data['OriginLocation'] = sample_df['OriginLocation'].values
    
    # Reorder to match features order before scaling
    imputed_data = imputed_data[features]
    
    # Apply scaling to numerical features
    numerical_cols = ['CulmenDepth', 'CulmenLength', 'BodyMass', 'FlipperLength']
    imputed_data[numerical_cols] = scaler.transform(imputed_data[numerical_cols])
    
    # Return as numpy array in the correct order (features order)
    return imputed_data.values