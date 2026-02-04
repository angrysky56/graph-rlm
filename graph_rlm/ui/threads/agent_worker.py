from PyQt6.QtCore import QThread, pyqtSignal, QObject
import asyncio
import queue
import time
from typing import Optional

# Import Backend Core
from graph_rlm.backend.src.core.agent import Agent
from graph_rlm.backend.src.core.database import db

class AgentWorker(QThread):
    # Signals to UI
    thoughtCreated = pyqtSignal(dict)      # New node data
    thoughtUpdated = pyqtSignal(dict)      # Update node status/content
    linkCreated = pyqtSignal(dict)         # New edge
    logMessage = pyqtSignal(str, str)      # (Level, Message)
    chatMessage = pyqtSignal(str, str)     # (Role, Content)
    statusChanged = pyqtSignal(str)        # Status bar text
    initialLoadComplete = pyqtSignal()     # Signal when initial state is done
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_running = True
        self.agent = None
        self.input_queue = queue.Queue() # UI -> Agent commands

    def run(self):
        """
        Main Thread Loop.
        Initializes the Agent and polls for events.
        """

        # Load initial state from DB FIRST
        # This ensures UI is populated even if agent init is slow or fails
        self._load_initial_state()

        # Initialize Agent in the thread context
        try:
            self.statusChanged.emit("Initializing Agent...")
            self.agent = Agent()

            # Run async initialization synchronously
            # Wrap in try/except to prevent blocking on network errors
            try:
                # Set a timeout for initialization
                # We can't easily timeout asyncio.run without wrapping the coroutine
                async def init_with_timeout():
                    try:
                        await asyncio.wait_for(self.agent.initialize_system(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logMessage.emit("WARNING", "Agent initialization timed out (Network slow/offline?). Proceeding.")
                    except Exception as e:
                        self.logMessage.emit("ERROR", f"Agent init error: {e}")

                asyncio.run(init_with_timeout())
                self.statusChanged.emit("Agent Ready")
            except Exception as e:
                self.logMessage.emit("ERROR", f"Agent init failed: {e}")
                self.statusChanged.emit("Agent Error")

        except Exception as e:
            self.logMessage.emit("ERROR", f"Failed to instantiate Agent: {e}")
            self.statusChanged.emit("Agent Error")
            # Don't return, keep running so UI stays responsive to logs/graph interactions

        # Start Event Polling Loop
        while self._is_running:
            # 1. Process Input Commands (Non-blocking)
            try:
                cmd_type, payload = self.input_queue.get_nowait()
                if cmd_type == "QUERY":
                    if self.agent:
                        self._handle_query(payload)
                    else:
                        self.logMessage.emit("ERROR", "Agent not initialized.")
                elif cmd_type == "STOP":
                    if self.agent:
                        self.agent.stop()
            except queue.Empty:
                pass

            # Idle sleep
            time.sleep(0.05)

    def _load_initial_state(self):
        """Loads existing graph state from the database and emits signals."""
        try:
            self.statusChanged.emit("Loading Graph...")
            nodes = db.get_graph_state() # Returns list of node dicts

            count = 0
            edge_count = 0
            # 1. Emit Nodes
            for node in nodes:
                # Ensure minimal data
                if "id" in node:
                    self.thoughtCreated.emit(node)
                    count += 1

                    # 2. Edges
                    pid = db.get_parent_id(node["id"])
                    if pid:
                        self.linkCreated.emit({"source": pid, "target": node["id"]})
                        edge_count += 1

            # Small sleep to let signal queue drain
            self.msleep(200)
            self.initialLoadComplete.emit()
            self.logMessage.emit("INFO", f"Loaded {count} nodes and {edge_count} edges from persistent memory.")

        except Exception as e:
            self.logMessage.emit("ERROR", f"Failed to load initial graph: {e}")

    def _handle_query(self, prompt: str):
        """
        Runs the async agent query in a blocking way (for this thread),
        streaming events back to Qt signals.
        """
        self.statusChanged.emit("Thinking...")
        self.chatMessage.emit("user", prompt)

        async def run_query():
            try:
                # Use the streaming interface
                async for event in self.agent.stream_query(prompt, session_id="ui_session"):
                    self._dispatch_event(event)
            except Exception as e:
                self.logMessage.emit("ERROR", f"Query Execution Error: {e}")

        # Run the async loop
        try:
            asyncio.run(run_query())
        except Exception as e:
            self.logMessage.emit("ERROR", f"Async Loop Error: {e}")

        self.statusChanged.emit("Idle")

    def _dispatch_event(self, event: dict):
        """Maps backend events to Qt Signals"""
        e_type = event.get("type")
        content = event.get("content", "")
        data = event.get("data", {})

        if e_type == "graph_update":
            action = data.get("action")
            if action == "add_node":
                self.thoughtCreated.emit(data.get("node"))
            elif action == "add_link":
                self.linkCreated.emit(data.get("link"))

        elif e_type == "thinking":
            self.logMessage.emit("INFO", content)

        elif e_type == "code_output_chunk":
            # Can direct this to a specific console widget if needed
            self.logMessage.emit("DEBUG", f"[CODE] {content}")

        elif e_type == "error":
            self.logMessage.emit("ERROR", content)

        elif e_type == "answer" or e_type == "RLM_FINAL_RESPONSE":
            self.chatMessage.emit("assistant", content)

        elif e_type == "scratchpad_update":
            # TODO: Emit signal to update scratchpad view
            pass

    def send_query(self, prompt: str):
        self.input_queue.put(("QUERY", prompt))

    def stop(self):
        self._is_running = False
        self.input_queue.put(("STOP", None))
