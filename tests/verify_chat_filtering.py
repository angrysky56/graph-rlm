import contextvars
import queue
import unittest

from graph_rlm.backend.src.core.state import broadcast_trace, execution_events


class TestChatFiltering(unittest.TestCase):
    def setUp(self):
        # Setup the context var with a queue
        self.q = queue.Queue()
        self.token = execution_events.set(self.q)

    def tearDown(self):
        execution_events.reset(self.token)

    def test_block_list(self):
        # Test that internal traces are blocked from chat
        broadcast_trace("[LLM] Generating response...")
        item = self.q.get()
        self.assertEqual(item["ui_target"], "TERMINAL_RAW")

        broadcast_trace("[DB] Cypher query executed")
        item = self.q.get()
        self.assertEqual(item["ui_target"], "TERMINAL_RAW")

    def test_meta_routing(self):
        # Test [META] routing
        broadcast_trace("[META] Analyzing step complexity...")
        item = self.q.get()
        self.assertEqual(item["ui_target"], "CHAT_RESPONSE")
        self.assertEqual(item["ui_component"], "meta_box")

    def test_reflexion_routing(self):
        # Test [REFLEXION] routing
        broadcast_trace("[REFLEXION] Detect loop in reasoning")
        item = self.q.get()
        self.assertEqual(item["ui_target"], "CHAT_RESPONSE")
        self.assertEqual(item["ui_component"], "reflexion_box")

    def test_agent_signal_vs_noise(self):
        # Test Agent noise filtering
        broadcast_trace("[AGENT] Thinking about life...") # Noise
        item = self.q.get()
        self.assertEqual(item["ui_target"], "TERMINAL_RAW")

        broadcast_trace("[AGENT] Decision: Proceed with plan") # Signal
        item = self.q.get()
        self.assertEqual(item["ui_target"], "CHAT_RESPONSE")
        self.assertEqual(item["ui_component"], "text")

if __name__ == "__main__":
    unittest.main()
