import torch
from model import MultiClassIDS, MULTICLASS_CLASS_NAMES
from server.aggregator import fed_avg

class TestMultiClassIDS:
    def test_output_shape(self):
        model = MultiClassIDS()
        x = torch.randn(32, 41)
        out = model(x)
        assert out.shape == (32, 5)

    def test_contract(self):
        model = MultiClassIDS()
        assert model.input_dim == 41
        assert model.num_classes == 5
        assert MULTICLASS_CLASS_NAMES == [
            "Normal", "DoS", "Probe", "R2L", "U2R"
        ]

    def test_get_set_weights(self):
        model = MultiClassIDS()
        weights = model.get_weights()
        model2 = MultiClassIDS()
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
