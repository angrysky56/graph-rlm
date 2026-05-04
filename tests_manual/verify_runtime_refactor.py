import unittest
from threading import Event

from graph_rlm.backend.src.mcp_integration import runtime


class TestRuntimeState(unittest.TestCase):
    def test_stop_event_management(self):
        """Verify that get/set_stop_event works with the new state container."""
        # 1. Verify initial state is None
        # We need to access the private state carefully for testing, or rely on public getters
        # Since usage is global, we should be careful not to break other tests if run in suite
        # runtime._state.stop_event = None  # Reset for test

        # 2. Set an event
        event = Event()
        runtime.set_stop_event(event)

        # 3. Get the event
        retrieved_event = runtime.get_stop_event()
        self.assertEqual(
            event, retrieved_event, "Retrieved event should match set event"
        )
        self.assertIsInstance(retrieved_event, Event)

        # 4. Verify uniqueness (create new event)
        new_event = Event()
        runtime.set_stop_event(new_event)
        self.assertEqual(runtime.get_stop_event(), new_event)
        self.assertNotEqual(runtime.get_stop_event(), event)


if __name__ == "__main__":
    unittest.main()
