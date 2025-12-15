import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_and_preprocess_data(debug=True):
    data = pd.read_csv(r'penguins.csv')
    if debug: print("Original data loaded:")
    if debug: print(data.head())

    # Encode OriginLocation
    origin_location_encoder = LabelEncoder()
    data['OriginLocation'] = origin_location_encoder.fit_transform(data['OriginLocation'])
    if debug: print("\nOriginLocation encoded:")
    if debug: print(data[['OriginLocation']].head())

    # Define columns - use consistent order
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']  # Keep this order for imputation
    median_col = ['BodyMass']

    mean_imputer = SimpleImputer(strategy='mean')
    median_imputer = SimpleImputer(strategy='median')

    # Apply imputation
    data[mean_cols] = mean_imputer.fit_transform(data[mean_cols])
    data[median_col] = median_imputer.fit_transform(data[median_col])

    if debug:
        print("\nData after imputation:")
        print(data.head())

    # Scale with CORRECT order to match features
    scaler = MinMaxScaler()
    # Use the exact same order as your features list
    numerical_cols_for_scaling = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
    data[numerical_cols_for_scaling] = scaler.fit_transform(data[numerical_cols_for_scaling])

    if debug:
        print("\nData after scaling:")
        print(data.head())
        print("Features after all preprocessing:")
        print(data[['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']].head())

    return data, origin_location_encoder, mean_imputer, median_imputer, scaler

def prepare_data_for_nn(debug=True):
    d, origin_location_encoder, mean_imputer, median_imputer, scaler = load_and_preprocess_data(debug=debug)

    label_encode = LabelEncoder()
    d['Species'] = label_encode.fit_transform(d['Species'])
    if debug:
        print("\nSpecies encoded:")
        print(d[['Species']].head())

    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    x, y = d[features].values, d['Species'].values
    if debug:
        print(f"\nFeatures order: {features}")
        print(f"First 5 samples:\n{x[:5]}")
        print(f"Labels:\n{y[:5]}")

    return x, y, label_encode, features, origin_location_encoder, mean_imputer, median_imputer, scaler


def split_data_by_class(x, y, train_samples=30, test_samples=20, debug=True):
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

    x_train = np.vstack(x_train_parts)
    y_train = np.hstack(y_train_parts)
    x_test = np.vstack(x_test_parts)
    y_test = np.hstack(y_test_parts)

    if debug:
        print(f"\nTraining set: {x_train.shape[0]} samples")
        print(f"Test set: {x_test.shape[0]} samples")
        print(f"Classes in training: {np.unique(y_train)}")
        print(f"First training sample:\n{x_train[0]}")
        print(f"First test sample:\n{x_test[0]}")

    return x_train, y_train, x_test, y_test


def one_hot_encode(y, num_classes, debug=True):
    y = y.astype(int)
    encoded = np.eye(num_classes)[y]
    if debug:
        print(f"\nOne-hot encoding of first 5 labels:\n{encoded[:5]}")
    return encoded


def preprocess_sample(sample_dict, features, origin_location_encoder, mean_imputer, median_imputer, scaler):
    # Create a DataFrame from the sample_dict with the CORRECT feature order
    sample_df = pd.DataFrame([sample_dict])

    # Encode OriginLocation
    if 'OriginLocation' in sample_df.columns:
        sample_df['OriginLocation'] = origin_location_encoder.transform(sample_df['OriginLocation'])

    # REORDER FIRST to match training feature order
    sample_df = sample_df[features]

    # Apply imputation
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']

    sample_df[mean_cols] = mean_imputer.transform(sample_df[mean_cols])
    sample_df[median_col] = median_imputer.transform(sample_df[median_col])

    # Apply scaling - use the exact same column order as training
    numerical_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']  # Match training order!
    sample_df[numerical_cols] = scaler.transform(sample_df[numerical_cols])

    # Final check to ensure correct order
    sample_df = sample_df[features]

    print("Final preprocessed sample:")
    print(sample_df)
    print(f"Final feature order: {sample_df.columns.tolist()}")

    return sample_df.values