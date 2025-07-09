import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)          # 실제 값, float으로 고정
        self.grad = 0.0       # 미분값(역전파로 채워짐)
        self._backward = lambda: None  # 역전파 함수
        self._prev = set(_children)    # 연산 그래프 상, 이 노드의 부모 노드들 
        self._op = _op            # 연산 종류 ex) '+', '*', '-', '/', '**', 'exp', 'log', 'log10', 'sin', 'cos', 'tanh', 'relu', 'sigmoid'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += out.grad / other.data
            other.grad -= (self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        if isinstance(other, Value):
            raise NotImplementedError("Power with another Value is not implemented")
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Value(-self.data, (self,), '-')
        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def log10(self):
        out = Value(math.log10(self.data), (self,), 'log10')
        def _backward():
            self.grad += (1 / (self.data * math.log(10))) * out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), 'cos')
        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __rsub__(self, other):
        return Value(other) - self
    def __rtruediv__(self, other):
        return Value(other) / self



    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
