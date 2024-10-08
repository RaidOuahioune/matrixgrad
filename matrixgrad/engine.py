import numpy as np
from graphviz import Digraph


class Tensor:
    """A class to represent a tensor and handle autograd for supported operations.
    Gradient computation is based on the chain rule, and the gradients are accumulated in the .grad attribute of each tensor.
    The gradients are computed using multiplication of global gradient with the local gradient of the operation.
    The gradients are accumulated in the positive direction of the function to maximize it.
    """

    def __init__(self, data, _children=(), _op=""):

        if isinstance(data, Tensor):
            print("Warning: Tensor object passed to Tensor constructor")
            data = data.data

        if not isinstance(data, np.ndarray):
            if isinstance(data, (float, int)):
                data = np.array([[data]], dtype=np.float64)
            else:
                data = np.array(data, dtype=np.float64)

        # Ensure data is at least 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape == (1, 1) or data.shape == (1,):
            data = data.reshape(-1, 1)

        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float64)

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self.prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):

        if isinstance(other, Tensor) and other.data.shape == self.data.shape:
            out = Tensor(self.data * other.data, (self, other), f"*")

            def _backward():
                self.grad += other.data * out.grad

                other.grad += self.data * out.grad

        else:
            if isinstance(other, Tensor):
                other = other.data.flatten()
            out = Tensor(self.data * other, (self,), f"*{other}")

            def _backward():
                self.grad += (other * out.grad).astype(np.float64)

        out._backward = _backward
        return out

    def sum(self):
        # Create a tensor with the sum of all elements in self.data

        out = Tensor(self.data.sum(), (self,), "sum")

        # Backward function to propagate the gradient
        def _backward():
            self.grad += np.ones_like(self.data, dtype=np.float64) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():

            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):

        out = Tensor(np.maximum(self.data, 0), (self,), "ReLU")

        def _backward():
            mask = (out.data > 0).reshape(out.grad.shape)
            result = np.multiply(mask, out.grad)

            self.grad += result

        out._backward = _backward

        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "sigmoid")

        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)

        out._backward = _backward
        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float64)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def _trace(self):
        """Recursively build a directed graph starting from the current node and moving backwards"""
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)

                for child in v.prev:
                    edges.add((child, v))
                    build(child)

        build(self)
        return nodes, edges

    def visualize(self, format="svg", rankdir="LR"):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ["LR", "TB"]
        nodes, edges = self._trace()
        dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

        for n in nodes:
            dot.node(
                name=str(id(n)),
                label=f"{n.data.shape}",
                shape="record",
            )
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
