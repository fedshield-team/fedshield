import torch
import numpy as np
import shap
import json
from model import IntrusionDetector

# Load model and data
model = IntrusionDetector()
model.load_state_dict(torch.load("models/federated_model.pth"))
model.eval()

X_test = np.load("data/X_test.npy")

FEATURE_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

# Use 100 background samples for SHAP
background = torch.FloatTensor(X_test[:100])
test_samples = torch.FloatTensor(X_test[100:200])

def model_predict(x):
    x_tensor = torch.FloatTensor(x)
    with torch.no_grad():
        return model(x_tensor).numpy()

print("Computing SHAP values... (takes ~1 min)")
explainer = shap.KernelExplainer(model_predict, background.numpy())
shap_values = explainer.shap_values(test_samples.numpy(), nsamples=100)

# Top 10 most important features
mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()
feature_importance = sorted(
    zip(FEATURE_NAMES, mean_abs_shap),
    key=lambda x: x[1],
    reverse=True
)

print("\n===== TOP 10 FEATURES DRIVING ATTACK DETECTION =====")
for i, (feat, score) in enumerate(feature_importance[:10]):
    print(f"{i+1:2d}. {feat:35s} SHAP: {score:.4f}")

# Save results
results = {
    "feature_importance": [
        {"feature": f, "shap_score": float(s)} 
        for f, s in feature_importance
    ]
}
with open("models/shap_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSHAP results saved to models/shap_results.json")
print("This explains WHY the model flags packets as attacks.")