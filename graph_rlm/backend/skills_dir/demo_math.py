"""
Demo Math Skill for Graph RLM.
This skill provides basic mathematical operations to demonstrate the skill system.
"""

def fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number efficiently.
    Args:
        n: The position in the sequence (0-based)
    Returns:
        The nth Fibonacci number
    """
    if n <= 0: return 0
    elif n == 1: return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n: int) -> bool:
    """
    Checks if a number is prime.
    """
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
