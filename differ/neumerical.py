def numerical_differ(func, x:float, h:float=1e-4) -> float:
    return (func(x + h) - func(x - h)) / (2 * h)
