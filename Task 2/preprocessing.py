import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

def preprocess_sample(sample_dict, features, origin_location_encoder, mean_imputer, median_imputer, scaler):
    df_sample = pd.DataFrame([sample_dict])
    for feature in features:
        if feature not in df_sample.columns:
            df_sample[feature] = None

    df_sample = df_sample[features]
    
    if 'OriginLocation' in df_sample.columns:
        df_sample['OriginLocation'] = origin_location_encoder.transform(df_sample['OriginLocation'])
    
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
    median_col = ['BodyMass']
    
    if all(col in df_sample.columns for col in mean_cols):
        df_sample[mean_cols] = mean_imputer.transform(df_sample[mean_cols])
    if all(col in df_sample.columns for col in median_col):
        df_sample[median_col] = median_imputer.transform(df_sample[median_col])
    
    scale_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength', 'BodyMass']
    if all(col in df_sample.columns for col in scale_cols):
        df_sample[scale_cols] = scaler.transform(df_sample[scale_cols])

    return df_sample[features].values

<<<<<<< HEAD

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
=======
def load_and_preprocess_data():
    d = pd.read_csv(r'penguins.csv')
    origin_location_encoder = LabelEncoder()
    origin_location_encoder.fit(d['OriginLocation'])
    mean_cols = ['CulmenDepth', 'CulmenLength', 'FlipperLength']
>>>>>>> c7e9de65bc8d168a8460bc1bc47492cfae821129
    median_col = ['BodyMass']

    mean_imputer = SimpleImputer(strategy='mean')
    mean_imputer.fit(d[mean_cols])
    median_imputer = SimpleImputer(strategy='median')
<<<<<<< HEAD

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
=======
    median_imputer.fit(d[median_col])
    
    tmp_mean_imputed = mean_imputer.transform(d[mean_cols])
    tmp_median_imputed = median_imputer.transform(d[median_col])
    tmp_imputed_df = pd.DataFrame(
        np.hstack([tmp_mean_imputed, tmp_median_imputed]),
        columns=mean_cols + median_col
    )
    
    scaler = MinMaxScaler()
    scaler.fit(tmp_imputed_df)
    
    feature_cols = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    
    prep_samples = []
    for idx, row in d.iterrows():
        sample_dict = {col: row[col] for col in feature_cols}
        prep_sample = preprocess_sample(
            sample_dict,
            feature_cols,
            origin_location_encoder,
            mean_imputer,
            median_imputer,
            scaler
        )
        prep_samples.append(prep_sample[0])
    
    prep_df = pd.DataFrame(
        prep_samples,
        columns=feature_cols
    )
    
    d[feature_cols] = prep_df

    return d, origin_location_encoder, mean_imputer, median_imputer, scaler
>>>>>>> c7e9de65bc8d168a8460bc1bc47492cfae821129

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
<<<<<<< HEAD

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

=======
    x_train, y_train, x_test, y_test = np.vstack(x_train_parts), np.hstack(y_train_parts), np.vstack(x_test_parts), np.hstack(y_test_parts)
    
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    
    print(f"training set: {x_train.shape[0]} samples")
    print(f"test set: {x_test.shape[0]} samples")
    print(f"Classes in training: {np.unique(y_train)}")
>>>>>>> c7e9de65bc8d168a8460bc1bc47492cfae821129
    return x_train, y_train, x_test, y_test


def one_hot_encode(y, num_classes, debug=True):
    y = y.astype(int)
<<<<<<< HEAD
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
=======
    return np.eye(num_classes)[y]
>>>>>>> c7e9de65bc8d168a8460bc1bc47492cfae821129
