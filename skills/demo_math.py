def demo_math(operation="fibonacci", n=10, **kwargs):
    """
    Demo Math Skill for Graph RLM.
    Supported operations: fibonacci, factorial, prime_check
    """

    def fibonacci(num):
        a, b = 0, 1
        result = []
        for _ in range(num):
            result.append(a)
            a, b = b, a + b
        return result

    def factorial(num):
        if num == 0:
            return 1
        res = 1
        for i in range(1, num + 1):
            res *= i
        return res

    # Logic to handle different operations
    op = operation.lower()
    # Ensure n is an integer
    val = int(n)

    if op == "fibonacci":
        return {"operation": op, "input": val, "result": fibonacci(val)}
    elif op == "factorial":
        return {"operation": op, "input": val, "result": factorial(val)}
    else:
        return {"error": f"Operation {op} not supported"}
