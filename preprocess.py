import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

def load_and_preprocess(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df.drop('difficulty', axis=1, inplace=True)
    
    # Encode categorical columns
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Binary classification: normal vs attack
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/KDDTrain+.txt")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Attack ratio in train: {y_train.mean():.2%}")
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    print("Preprocessed data saved!")