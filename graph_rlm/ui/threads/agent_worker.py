from PyQt6.QtCore import QThread, pyqtSignal, QObject
import asyncio
import queue
import time
from typing import Optional

# Import Backend Core
from graph_rlm.backend.src.core.agent import Agent

class AgentWorker(QThread):
    # Signals to UI
    thoughtCreated = pyqtSignal(dict)      # New node data
    thoughtUpdated = pyqtSignal(dict)      # Update node status/content
    linkCreated = pyqtSignal(dict)         # New edge
    logMessage = pyqtSignal(str, str)      # (Level, Message)
    chatMessage = pyqtSignal(str, str)     # (Role, Content)
    statusChanged = pyqtSignal(str)        # Status bar text
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
        # Initialize Agent in the thread context
        try:
            self.statusChanged.emit("Initializing Agent...")
            self.agent = Agent()

            # Run async initialization synchronously
            asyncio.run(self.agent.initialize_system())

            self.statusChanged.emit("Agent Ready")
        except Exception as e:
            self.logMessage.emit("ERROR", f"Failed to initialize Agent: {e}")
            return

        # Start Event Polling Loop
        while self._is_running:
            # 1. Process Input Commands (Non-blocking)
            try:
                cmd_type, payload = self.input_queue.get_nowait()
                if cmd_type == "QUERY":
                    self._handle_query(payload)
                elif cmd_type == "STOP":
                    if self.agent:
                        self.agent.stop()
            except queue.Empty:
                pass

            # 2. Process Agent Events (Poll Queue)
            # The agent uses a context var queue, but since we are running everything
            # in this thread (or launching tasks from it), we need to bridge the events.
            # However, `agent.stream_query` is an async generator.
            # We wrapped that logic in `_handle_query` which runs an asyncio loop.

            # Idle sleep
            time.sleep(0.05)

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
