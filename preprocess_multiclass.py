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

# Standard NSL-KDD 5-class taxonomy
ATTACK_MAP = {
    'normal': 0,
    # DoS
    'neptune': 1, 'back': 1, 'teardrop': 1, 'pod': 1, 
    'smurf': 1, 'land': 1, 'mailbomb': 1, 'apache2': 1,
    # Probe
    'satan': 2, 'ipsweep': 2, 'portsweep': 2, 'nmap': 2, 'mscan': 2,
    # R2L
    'warezclient': 3, 'warezmaster': 3, 'imap': 3, 'ftp_write': 3,
    'multihop': 3, 'guess_passwd': 3, 'phf': 3, 'spy': 3, 'sendmail': 3,
    # U2R
    'buffer_overflow': 4, 'rootkit': 4, 'loadmodule': 4, 'perl': 4, 'xterm': 4
}

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

def load_multiclass(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df.drop('difficulty', axis=1, inplace=True)
    
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Map to 5 classes, drop unknown
    df['label'] = df['label'].map(ATTACK_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (df['label'] == i).sum()
        print(f"  {name}: {count} samples")
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_multiclass("data/KDDTrain+.txt")
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    np.save("data/X_train_mc.npy", X_train)
    np.save("data/X_test_mc.npy", X_test)
    np.save("data/y_train_mc.npy", y_train)
    np.save("data/y_test_mc.npy", y_test)
    print("Saved multiclass data!")