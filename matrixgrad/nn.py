import numpy as np
from matrixgrad.engine import Tensor


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self) -> list[Tensor]:
        return []


class Layer(Module):

    def __init__(self, input_dim, output_dim, nonlin=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = Tensor(np.random.uniform(-1, 1, (input_dim, output_dim)))

        self.nonlin = nonlin

    def __call__(self, x):
        act: Tensor = x @ self.w
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Tensor]:
        return [self.w]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Layer({self.input_dim}, {self.output_dim})"


class MLP(Module):

    def __init__(self, layer_shapes: list[int], nouts):
        """
        layer_shapes: list of integers, the number of nodes in each layer(including input layer)\n

        nouts: the number of output nodes
        """
        self.layers = [
            Layer(nin, nout, nonlin=True)
            for nin, nout in zip(layer_shapes, layer_shapes[1:])
        ]
        self.layers.append(Layer(layer_shapes[-1], nouts, nonlin=False))

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def num_of_parameters(self):
        return sum(
            np.prod(p.data.shape) for l in self.layers for p in l.parameters()[0:1]
        )  # + len( self.layers # the biases

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
