import torch
import copy

def fed_avg(weights_list):
    """
    FedAvg algorithm — average weights from all nodes.
    weights_list: list of weight lists from each node
    Returns: averaged weights
    """
    avg_weights = []
    for layer_idx in range(len(weights_list[0])):
        layer_weights = torch.stack([
            weights_list[i][layer_idx] 
            for i in range(len(weights_list))
        ])
        avg_weights.append(layer_weights.mean(dim=0))
    return avg_weights