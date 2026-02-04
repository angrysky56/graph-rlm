from .client import client
from .operations import *
from .search import *

# Facade class to maintain backward compatibility with 'db.db' or 'agent.db' access patterns
class GraphClientFacade:
    def __init__(self):
        self.client = client

    def __getattr__(self, name):
        # Delegate attributes to the client instance first
        if hasattr(self.client, name):
            return getattr(self.client, name)
        # Then check global operations module (since we imported * from operations)
        if name in globals():
            return globals()[name]
        raise AttributeError(f"GraphClientFacade has no attribute '{name}'")

    # Explicitly map common methods to global functions for IDE support/clarity
    def create_thought_node(self, *args, **kwargs): return create_thought_node(*args, **kwargs)
    def get_graph_state(self, *args, **kwargs): return get_graph_state(*args, **kwargs)
    def get_parent_id(self, *args, **kwargs): return get_parent_id(*args, **kwargs)
    def delete_thought_node(self, *args, **kwargs): return delete_thought_node(*args, **kwargs)
    def update_thought_result(self, *args, **kwargs): return update_thought_result(*args, **kwargs)
    def find_similar_thoughts(self, *args, **kwargs): return find_similar_thoughts(*args, **kwargs)
    def create_vector_indexes(self, *args, **kwargs): return create_vector_indexes(*args, **kwargs)
    def get_context_frontier(self, *args, **kwargs): return get_context_frontier(*args, **kwargs)
    def reembed_all_thoughts(self, *args, **kwargs): return reembed_all_thoughts(*args, **kwargs)
    def save_round(self, *args, **kwargs): return save_round(*args, **kwargs)
    def get_completed_rounds(self, *args, **kwargs): return get_completed_rounds(*args, **kwargs)
    def get_session_trace(self, *args, **kwargs): return get_session_trace(*args, **kwargs)
    def delete_session(self, *args, **kwargs): return delete_session(*args, **kwargs)
    def prune_orphans(self, *args, **kwargs): return prune_orphans(*args, **kwargs)
    def reset_graph(self, *args, **kwargs): return reset_graph(*args, **kwargs)
    def mark_nodes_as_consolidated(self, *args, **kwargs): return mark_nodes_as_consolidated(*args, **kwargs)
    def perform_synaptic_homeostasis(self, *args, **kwargs): return perform_synaptic_homeostasis(*args, **kwargs)

    # Query override
    def query(self, *args, **kwargs): return self.client.query(*args, **kwargs)

db = GraphClientFacade()
