import pytest
import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import IntrusionDetector
from server.aggregator import fed_avg
from nodes.node import FedNode

class TestIntrusionDetector:
    def test_output_shape(self):
        model = IntrusionDetector()
        x = torch.randn(32, 41)
        out = model(x)
        assert out.shape == (32, 1)

    def test_output_range(self):
        model = IntrusionDetector()
        x = torch.randn(100, 41)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_get_set_weights(self):
        model = IntrusionDetector()
        weights = model.get_weights()
        model2 = IntrusionDetector()
        model2.set_weights(weights)
        for w1, w2 in zip(model.get_weights(), model2.get_weights()):
            assert torch.allclose(w1, w2)

class TestFedAvg:
    def test_averaging(self):
        w1 = [torch.ones(10, 5), torch.ones(10)]
        w2 = [torch.zeros(10, 5), torch.zeros(10)]
        avg = fed_avg([w1, w2])
        expected = torch.full((10, 5), 0.5)
        assert torch.allclose(avg[0], expected)

    def test_three_nodes(self):
        weights = [[torch.randn(8, 4) for _ in range(3)] for _ in range(3)]
        avg = fed_avg(weights)
        assert len(avg) == 3

class TestFedNode:
    def test_init(self):
        X = np.random.randn(200, 41)
        y = np.random.randint(0, 2, 200).astype(float)
        node = FedNode(1, X, y)
        assert node.node_id == 1

    def test_weights_exchange(self):
        X = np.random.randn(200, 41)
        y = np.random.randint(0, 2, 200).astype(float)
        node1 = FedNode(1, X, y)
        node2 = FedNode(2, X, y)
        weights = node1.get_weights()
        node2.set_weights(weights)
        for w1, w2 in zip(node1.get_weights(), node2.get_weights()):
            assert torch.allclose(w1, w2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])