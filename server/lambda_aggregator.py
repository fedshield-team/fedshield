import json
import numpy as np


def fed_avg_numpy(
    weights_list,
    sample_counts=None
):
    """
    NumPy implementation of weighted FedAvg.

    weights_list:
        [
            [layer1, layer2, ...],   # client 1
            [layer1, layer2, ...],   # client 2
            ...
        ]

    sample_counts:
        Number of samples belonging to each client.
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
            "sample_counts must match "
            "number of clients"
        )

    if any(
        count <= 0
        for count in sample_counts
    ):
        raise ValueError(
            "All sample counts must be positive"
        )

    num_layers = len(
        weights_list[0]
    )

    for client_weights in weights_list:

        if len(client_weights) != num_layers:
            raise ValueError(
                "All clients must provide "
                "the same number of layers"
            )

    total_samples = float(
        sum(sample_counts)
    )

    averaged_weights = []

    for layer_idx in range(num_layers):

        layer = np.zeros_like(
            np.asarray(
                weights_list[0][layer_idx],
                dtype=np.float32
            )
        )

        for client_idx in range(num_clients):

            client_weight = (
                sample_counts[client_idx]
                / total_samples
            )

            layer += (
                np.asarray(
                    weights_list[client_idx][layer_idx],
                    dtype=np.float32
                )
                * client_weight
            )

        averaged_weights.append(
            layer.tolist()
        )

    return averaged_weights


def lambda_handler(event, context):

    try:

        if isinstance(
            event.get("body"),
            str
        ):
            body = json.loads(
                event["body"]
            )
        else:
            body = event

        weights_list = body.get(
            "weights",
            []
        )

        sample_counts = body.get(
            "sample_counts"
        )

        if len(weights_list) < 2:

            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error":
                            "Need at least 2 clients"
                    }
                )
            }

        if sample_counts is not None:

            if len(sample_counts) != len(
                weights_list
            ):

                return {
                    "statusCode": 400,
                    "body": json.dumps(
                        {
                            "error":
                                "sample_counts must match "
                                "number of clients"
                        }
                    )
                }

        print(
            f"Aggregating weights from "
            f"{len(weights_list)} clients..."
        )

        averaged = fed_avg_numpy(
            weights_list,
            sample_counts
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "averaged_weights":
                        averaged,

                    "num_nodes":
                        len(weights_list),

                    "sample_counts":
                        sample_counts,

                    "message":
                        "FedAvg aggregation complete"
                }
            )
        }

    except Exception as exc:

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error":
                        str(exc)
                }
            )
        }


if __name__ == "__main__":

    import os
    import sys
    import torch

    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    from model import IntrusionDetector

    model = IntrusionDetector()

    weights = [
        [
            parameter.detach()
            .cpu()
            .numpy()
            .tolist()
            for parameter in model.parameters()
        ]
        for _ in range(3)
    ]

    test_event = {
        "weights": weights,
        "sample_counts": [
            100,
            200,
            300
        ]
    }

    result = lambda_handler(
        test_event,
        None
    )

    body = json.loads(
        result["body"]
    )

    print(
        body["message"]
    )

    print(
        f"Aggregated "
        f"{body['num_nodes']} clients successfully"
    )