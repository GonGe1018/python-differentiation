from differ.autodiff import Value

def poly_example():
    # 다항 함수의 미분 예제
    val = 2.0
    x = Value(val)
    
    f = x**2 - x
    f.backward()
    
    print("Polynomial Example:")
    print("f(x) = x**2 - x")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}") 
    print("\n")

def exponential_example():
    # 지수 함수의 미분 예제
    val = 2.0
    x = Value(val)
    
    f = x.exp() + x.log()
    f.backward()
    
    print("Exponential Example:")
    print("f(x) = exp(x) + log(x)")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")

def logarithmic_example():
    # 로그 함수의 미분 예제
    val = 10.0
    x = Value(val)
    
    f = x.log() + x.log10()
    f.backward()
    
    print("Logarithmic Example:")
    print("f(x) = log(x) + log10(x)")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")

def trig_example():
    # 삼각 함수의 미분 예제
    val = 0.5  # 30도 (라디안 단위)
    x = Value(val)  
    
    f = x.sin() + x.cos()
    f.backward()

    print("Trigonometric Example:")
    print("f(x) = sin(x) + cos(x)")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")

def relu_example():
    # ReLU 함수의 미분 예제
    val = 2.0
    x = Value(val)
    
    f = x.relu()
    f.backward()
    
    print("ReLU Example:")
    print("f(x) = relu(x)")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")

def sigmoid_example():
    # 시그모이드 함수의 미분 예제
    val = 0.5
    x = Value(val)
    
    f = x.sigmoid()
    f.backward()
    
    print("Sigmoid Example:")
    print("f(x) = sigmoid(x) = 1 / (1 + exp(-x))")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")


def multivariable_example():
    # 다변수 함수의 미분 예제
    x = Value(3.0)
    y = Value(4.0)
    
    f = x * y + x**3 - y.log()
    f.backward()
    
    print("Multivariable Example:")
    print("f(x, y) = x*y + x**3 - log(y)")
    print(f"x: {x.data}, df/dx: {x.grad}")
    print(f"y: {y.data}, df/dy: {y.grad}")
    print(f"f: {f.data}, df/df: {f.grad}")
    print("\n")

def main():
    poly_example()
    exponential_example()
    logarithmic_example()
    trig_example()
    relu_example()
    sigmoid_example()
    multivariable_example()
    

if __name__ == "__main__":
    main()
    
