class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data          # 실제 값
        self.grad = 0             # 미분값(역전파로 채워짐)
        self._backward = lambda: None  # 역전파 함수
        self._prev = set(_children)    # 이 노드의 부모 노드들 (연산 그래프)
        self._op = _op            # 연산 종류(+, *, ...)

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
