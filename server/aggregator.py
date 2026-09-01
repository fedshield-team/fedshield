import torch


def fed_avg(
    weights_list,
    sample_counts=None
):
    """
    Weighted Federated Averaging.

    Each client's contribution is weighted by
    its number of training samples.
    """

    if not weights_list:
        raise ValueError(
            "weights_list cannot be empty."
        )

    num_clients = len(
        weights_list
    )

    if sample_counts is None:

        sample_counts = [
            1
        ] * num_clients

    if len(sample_counts) != num_clients:

        raise ValueError(
            "weights_list and sample_counts "
            "must have the same length."
        )

    if any(
        count <= 0
        for count in sample_counts
    ):

        raise ValueError(
            "All sample counts must be positive."
        )

    num_layers = len(
        weights_list[0]
    )

    for client_weights in weights_list:

        if len(client_weights) != num_layers:

            raise ValueError(
                "All clients must provide "
                "the same number of tensors."
            )

    total_samples = float(
        sum(sample_counts)
    )

    averaged_weights = []

    for layer_idx in range(
        num_layers
    ):

        reference = (
            weights_list[0][layer_idx]
        )

        averaged_layer = torch.zeros_like(
            reference,
            dtype=torch.float32
        )

        for client_idx in range(
            num_clients
        ):

            weight = (
                sample_counts[client_idx]
                /
                total_samples
            )

            averaged_layer += (
                weights_list[
                    client_idx
                ][layer_idx]
                .to(dtype=torch.float32)
                * weight
            )

        averaged_weights.append(
            averaged_layer.to(
                dtype=reference.dtype
            )
        )

    return averaged_weights