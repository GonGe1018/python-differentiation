from auto_differ.value import Value

def main():
    x = Value(2.0)
    y = Value(3.0)
    
    f = x * y + x
    f.backward()
    
    print(f"x: {x.data}, grad: {x.grad}")
    print(f"y: {y.data}, grad: {y.grad}")
    print(f"f: {f.data}, grad: {f.grad}") 

if __name__ == "__main__":
    main()