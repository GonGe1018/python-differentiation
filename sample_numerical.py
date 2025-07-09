from differ.neumerical import numerical_differ
import math

def numerical_poly_example():
    # 다항 함수의 수치 미분 예제
    def poly_func(x):
        return x**2 - x

    val = 2.0
    grad = numerical_differ(poly_func, val)

    print("Numerical Polynomial Example:")
    print("f(x) = x**2 - x")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def numerical_exponential_example():
    # 지수 함수의 수치 미분 예제
    def exp_func(x):
        return math.exp(x) + math.log(x)

    val = 2.0
    grad = numerical_differ(exp_func, val)

    print("Numerical Exponential Example:")
    print("f(x) = exp(x) + log(x)")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def numerical_logarithmic_example():
    # 로그 함수의 수치 미분 예제
    def log_func(x):
        return math.log(x) + math.log10(x)

    val = 10.0
    grad = numerical_differ(log_func, val)

    print("Numerical Logarithmic Example:")
    print("f(x) = log(x) + log10(x)")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def numerical_trig_example():
    # 삼각 함수의 수치 미분 예제
    def trig_func(x):
        return math.sin(x) + math.cos(x)

    val = 0.5  # 30도 (라디안 단위)
    grad = numerical_differ(trig_func, val)

    print("Numerical Trigonometric Example:")
    print("f(x) = sin(x) + cos(x)")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def numerical_relu_example():
    # ReLU 함수의 수치 미분 예제
    def relu_func(x):
        return max(0, x)

    val = 2.0
    grad = numerical_differ(relu_func, val)

    print("Numerical ReLU Example:")
    print("f(x) = ReLU(x)")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def numerical_sigmoid_example():
    # Sigmoid 함수의 수치 미분 예제
    def sigmoid_func(x):
        return 1 / (1 + math.exp(-x))

    val = 0.0
    grad = numerical_differ(sigmoid_func, val)

    print("Numerical Sigmoid Example:")
    print("f(x) = sigmoid(x)")
    print(f"x: {val}, df/dx: {grad}")
    print("\n")

def main():
    numerical_poly_example()
    numerical_exponential_example()
    numerical_logarithmic_example()
    numerical_trig_example()
    numerical_relu_example()
    numerical_sigmoid_example()

if __name__ == "__main__":
    main()