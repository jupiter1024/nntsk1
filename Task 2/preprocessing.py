import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

def preprocess_sample(sample_dict, features, origin_location_encoder, mean_imputer, median_imputer, scaler):
    # Create DataFrame from sample
    df_sample = pd.DataFrame([sample_dict])
    
    # Ensure all features are present
    for feature in features:
        if feature not in df_sample.columns:
            df_sample[feature] = None
    
    # Reorder columns to match features order
    df_sample = df_sample[features]
    
    # Step 1: Encode OriginLocation
    if 'OriginLocation' in df_sample.columns:
        df_sample['OriginLocation'] = origin_location_encoder.transform(df_sample['OriginLocation'])
    
    # Step 2: Apply imputation
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']
    
    if all(col in df_sample.columns for col in mean_cols):
        df_sample[mean_cols] = mean_imputer.transform(df_sample[mean_cols])
    if all(col in df_sample.columns for col in median_col):
        df_sample[median_col] = median_imputer.transform(df_sample[median_col])
    
    # Step 3: Apply scaling (must match the order used during fit: mean_cols + median_col)
    scale_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength', 'BodyMass']
    if all(col in df_sample.columns for col in scale_cols):
        # Transform with DataFrame to match the fit (prevents feature name warning)
        df_sample[scale_cols] = scaler.transform(df_sample[scale_cols])
    
    # Return as numpy array in the correct feature order
    return df_sample[features].values

def load_and_preprocess_data():
    """
    Load data and fit all transformers, then preprocess all samples using preprocess_single_sample.
    """
    # Load raw data
    d = pd.read_csv(r'penguins.csv')
    
    # Fit transformers on full dataset
    origin_location_encoder = LabelEncoder()
    origin_location_encoder.fit(d['OriginLocation'])
    
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']
    
    mean_imputer = SimpleImputer(strategy='mean')
    mean_imputer.fit(d[mean_cols])
    
    median_imputer = SimpleImputer(strategy='median')
    median_imputer.fit(d[median_col])
    
    # Fit scaler (need to impute first for fitting)
    temp_mean_imputed = mean_imputer.transform(d[mean_cols])
    temp_median_imputed = median_imputer.transform(d[median_col])
    temp_imputed_df = pd.DataFrame(
        np.hstack([temp_mean_imputed, temp_median_imputed]),
        columns=mean_cols + median_col
    )
    
    scaler = MinMaxScaler()
    scaler.fit(temp_imputed_df)
    
    # Define feature columns
    feature_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    
    # Now preprocess each sample using the same function
    preprocessed_samples = []
    for idx, row in d.iterrows():
        # Create sample dict with only feature columns
        sample_dict = {col: row[col] for col in feature_cols}
        preprocessed_sample = preprocess_sample(
            sample_dict,
            feature_cols,
            origin_location_encoder,
            mean_imputer,
            median_imputer,
            scaler
        )
        preprocessed_samples.append(preprocessed_sample[0])  # Remove extra dimension
    
    # Reconstruct DataFrame with preprocessed data
    preprocessed_df = pd.DataFrame(
        preprocessed_samples,
        columns=feature_cols
    )
    
    # Update original dataframe with preprocessed values
    d[feature_cols] = preprocessed_df
    
    return d, origin_location_encoder, mean_imputer, median_imputer, scaler

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
    
    # Shuffle training data to prevent bias from data order
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    
    # Shuffle test data to prevent bias from data order
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    
    print(f"training set: {x_train.shape[0]} samples")
    print(f"test set: {x_test.shape[0]} samples")
    print(f"Classes in training: {np.unique(y_train)}")
    return x_train, y_train, x_test, y_test

def one_hot_encode(y, num_classes):
    y = y.astype(int)
    return np.eye(num_classes)[y]
