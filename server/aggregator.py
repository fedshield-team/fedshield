import torch


def fed_avg(weights_list, sample_counts=None):
    """
    Federated Averaging (FedAvg).

    Performs weighted averaging of client model parameters.

    If sample_counts are provided:

        weight_i = samples_i / total_samples

    Otherwise all clients have equal weight.
    """

    if not weights_list:
        raise ValueError(
            "weights_list cannot be empty"
        )

    num_clients = len(weights_list)

    if sample_counts is None:
        sample_counts = [1] * num_clients

    if len(sample_counts) != num_clients:
        raise ValueError(
            "weights_list and sample_counts "
            "must have the same length"
        )

    if any(
        count <= 0
        for count in sample_counts
    ):
        raise ValueError(
            "All sample counts must be positive"
        )

    # Make sure every client has the same number of tensors.
    num_layers = len(weights_list[0])

    for client_weights in weights_list:

        if len(client_weights) != num_layers:
            raise ValueError(
                "All clients must provide "
                "the same number of model tensors"
            )

    total_samples = float(
        sum(sample_counts)
    )

    averaged_weights = []

    for layer_idx in range(num_layers):

        reference = weights_list[0][layer_idx]

        averaged_layer = torch.zeros_like(
            reference,
            dtype=torch.float32
        )

        for client_idx in range(num_clients):

            client_weight = (
                sample_counts[client_idx]
                / total_samples
            )

            averaged_layer += (
                weights_list[client_idx][layer_idx]
                .to(dtype=torch.float32)
                * client_weight
            )

        # Return same dtype as original model.
        averaged_weights.append(
            averaged_layer.to(
                dtype=reference.dtype
            )
        )

    return averaged_weights