"""
Configuration for the Sheaf-Theoretic Navigator.
Hyperparameters for curiosity-driven exploration and topological analysis.
"""

# Compression Progress (R(t))
COMPRESSION_WINDOW = 5  # Number of steps to measure C(D, t)
LZMA_PRESET = 6  # Compression level (0-9)

# Causal Entropic Forces (S_tau)
ENTROPY_LOOKAHEAD = 3   # Depth of future state tree for S_tau
FUTURE_DISCOUNT = 0.9   # Discount factor for future freedom

# Curiosity Thresholds
CURIOSITY_THRESHOLD = 0.6  # Minimum score to trigger curiosity-driven override
BOREDOM_THRESHOLD = 0.1    # Score below which the agent should switch strategies

# Langton's Lambda (Edge of Chaos)
# Class 4 behavior typically found around lambda ~ 0.4 - 0.6
EDGE_OF_CHAOS_LAMBDA_MIN = 0.35
EDGE_OF_CHAOS_LAMBDA_MAX = 0.65

# Recursion Safety
MAX_CURIOSITY_DEPTH = 2  # Prevent infinite curiosity loops
