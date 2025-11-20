import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data():
    """Your working data loading function"""
    data = pd.read_csv(r'A:\Collage\HandsOn\NN Task\nntsk1\Task 2\penguins.csv')
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

def prepare_data_for_nn():
    """Prepare data specifically for neural network"""
    data = load_and_preprocess_data()
    
    # Encode species labels
    label_encoder = LabelEncoder()
    data['Species'] = label_encoder.fit_transform(data['Species'])
    
    # Select features
    features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']
    X = data[features].values
    y = data['Species'].values
    
    return X, y, label_encoder, features

def split_data_by_class(X, y, train_samples=30, test_samples=20):
    """Split data: 30 training, 20 testing per class"""
    unique_classes = np.unique(y)
    
    X_train_parts = []
    y_train_parts = []
    X_test_parts = []
    y_test_parts = []
    
    for class_label in unique_classes:
        # Get all samples for this class
        class_mask = (y == class_label)
        class_X = X[class_mask]
        class_y = y[class_mask]
        
        # Take first 30 for training, next 20 for testing
        X_train_parts.append(class_X[:train_samples])
        y_train_parts.append(class_y[:train_samples])
        X_test_parts.append(class_X[train_samples:train_samples + test_samples])
        y_test_parts.append(class_y[train_samples:train_samples + test_samples])
    
    # Combine all classes
    X_train = np.vstack(X_train_parts)
    y_train = np.hstack(y_train_parts)
    X_test = np.vstack(X_test_parts)
    y_test = np.hstack(y_test_parts)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes in training: {np.unique(y_train)}")
    
    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding"""
    # Ensure y is integer type
    y = y.astype(int)
    return np.eye(num_classes)[y]