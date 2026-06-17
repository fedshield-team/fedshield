import json
import numpy as np

def fed_avg_numpy(weights_list):
    """FedAvg on numpy arrays — runs on AWS Lambda"""
    avg_weights = []
    for layer_idx in range(len(weights_list[0])):
        layer = np.mean([
            np.array(weights_list[i][layer_idx]) 
            for i in range(len(weights_list))
        ], axis=0)
        avg_weights.append(layer.tolist())
    return avg_weights

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Expects: {"weights": [node1_weights, node2_weights, node3_weights]}
    Returns: {"averaged_weights": [...], "num_nodes": 3}
    """
    try:
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event
        
        weights_list = body.get("weights", [])
        
        if len(weights_list) < 2:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Need at least 2 nodes"})
            }
        
        print(f"Aggregating weights from {len(weights_list)} nodes...")
        averaged = fed_avg_numpy(weights_list)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "averaged_weights": averaged,
                "num_nodes": len(weights_list),
                "message": "FedAvg aggregation complete"
            })
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from model import IntrusionDetector
    
    model = IntrusionDetector()
    weights = [[p.data.numpy().tolist() for p in model.parameters()] for _ in range(3)]
    
    test_event = {"weights": weights}
    result = lambda_handler(test_event, None)
    print(json.loads(result["body"])["message"])
    print(f"Aggregated {json.loads(result['body'])['num_nodes']} nodes successfully")