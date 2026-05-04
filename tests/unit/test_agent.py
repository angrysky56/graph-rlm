"""Unit tests for src.core.agent module.
Tests for Agent class initialization and basic utility methods.
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from graph_rlm.backend.src.core.agent import (
    Agent,
    validate_agent_prompt,
    validate_session_id,
)
from graph_rlm.backend.src.core.exceptions import ValidationError


class TestAgentValidators:
    """Test validator functions in agent module."""

    def test_validate_agent_prompt_success(self):
        """Test valid prompt validation."""
        validate_agent_prompt("Valid prompt")

    def test_validate_agent_prompt_empty(self):
        """Test empty prompt validation."""
        with pytest.raises(ValidationError) as exc:
            validate_agent_prompt("")
        assert "cannot be empty" in str(exc.value)

    def test_validate_agent_prompt_too_long(self):
        """Test too long prompt validation."""
        with pytest.raises(ValidationError) as exc:
            validate_agent_prompt("a" * 101, max_length=100)
        assert "exceeds maximum length" in str(exc.value)

    def test_validate_session_id_success(self):
        """Test valid session ID validation."""
        validate_session_id("550e8400-e29b-41d4-a716-446655440000")

    def test_validate_session_id_invalid(self):
        """Test invalid session ID validation."""
        with pytest.raises(ValidationError) as exc:
            validate_session_id("not-a-uuid")
        assert "valid UUID" in str(exc.value)

    def test_validate_session_id_empty(self):
        """Test empty session ID validation."""
        with pytest.raises(ValidationError) as exc:
            validate_session_id("")
        assert "non-empty string" in str(exc.value)


@pytest.mark.asyncio
class TestAgentCore:
    """Test Agent core functionality."""

    async def test_agent_init(self):
        """Test Agent initialization."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"), \
             mock.patch("graph_rlm.backend.src.core.agent.settings") as mock_settings:
            
            mock_settings.KNOWLEDGE_BASE_PATH = "/tmp/kb_test"
            
            with mock.patch("pathlib.Path.mkdir"), \
                 mock.patch("pathlib.Path.write_text"), \
                 mock.patch("pathlib.Path.exists", return_value=False):
                
                agent = Agent()
                assert agent.db is not None
                assert agent.runtime is not None
                assert agent.navigator is not None
                assert agent.morph_memory is not None
                assert agent.stop_requested is False

    async def test_ensure_kb_structure_fail(self):
        """Test KB structure creation failure handling."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"), \
             mock.patch("graph_rlm.backend.src.core.agent.settings") as mock_settings:
            
            mock_settings.KNOWLEDGE_BASE_PATH = "/tmp/kb_test"
            
            # Mock Path to raise error
            with mock.patch("pathlib.Path.mkdir", side_effect=OSError("Forbidden")):
                agent = Agent()
                # Should not crash, just log warning
                assert agent.db is not None

    async def test_generate_axiom_search_query_success(self):
        """Test successful axiom query generation."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            
            with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", 
                           return_value="security, encryption"):
                query = await agent._generate_axiom_search_query("Help me encrypt a file")
                assert "security, encryption" in query
                assert "Help me encrypt a file" in query

    async def test_generate_axiom_search_query_error(self):
        """Test fallback when axiom query generation fails."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            
            with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", 
                           side_effect=ValueError("LLM Error")):
                query = await agent._generate_axiom_search_query("Test prompt")
                assert query == "Test prompt"

    async def test_handle_llm_circuit_open(self):
        """Test LLM circuit open handling."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            from graph_rlm.backend.src.core.circuit import CircuitOpenError
            error = CircuitOpenError("Circuit 'llm' is open", circuit_name="llm")
            
            result = await agent._handle_llm_circuit_open(error)
            assert "temporarily unavailable" in result
            agent.emit_event.assert_called_with("error", content=mock.ANY)

    async def test_session_properties(self):
        """Test session-isolated properties using ExecutionState."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"), \
             mock.patch("graph_rlm.backend.src.core.agent.agent_state") as mock_state:
            
            from graph_rlm.backend.src.core.state import ExecutionState
            state = ExecutionState()
            mock_state.get.return_value = state
            
            agent = Agent()
            
            agent.final_result = "Success"
            assert state.final_result == "Success"
            assert agent.final_result == "Success"
            
            agent.stop_requested = True
            assert state.stop_requested is True
            assert agent.stop_requested is True
            
            agent.synthesis_triggered = True
            assert state.synthesis_triggered is True
            assert agent.synthesis_triggered is True
            
            agent.current_thought_id = "thought-1"
            assert state.current_thought_id == "thought-1"
            assert agent.current_thought_id == "thought-1"
            
            agent.current_depth = 5
            assert state.depth == 5
            assert agent.current_depth == 5

    async def test_record_turn(self):
        """Test recording a turn."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.record_turn(10)
            assert agent.current_turn == 10

    async def test_install_to_active_env_success(self):
        """Test successful package installation to active env."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            mock_proc = mock.AsyncMock()
            mock_proc.communicate.return_value = (b"Success", b"")
            mock_proc.returncode = 0
            
            with mock.patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await agent._install_to_active_env("requests")
                assert "Successfully installed requests" in result
                agent.emit_event.assert_any_call("token", content=mock.ANY)

    async def test_install_to_agent_venv_success(self):
        """Test successful package installation to agent venv."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Success"
            
            with mock.patch("pathlib.Path.exists", return_value=True), \
                 mock.patch("subprocess.run", return_value=mock_result):
                result = agent._install_to_agent_venv("numpy")
                assert "Successfully installed numpy" in result

    async def test_read_skill_success(self):
        """Test successful skill reading."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            with mock.patch("graph_rlm.backend.src.core.agent.is_mcp_available", return_value=True), \
                 mock.patch("graph_rlm.backend.src.core.agent.is_skills_available", return_value=True), \
                 mock.patch("graph_rlm.backend.src.core.agent.get_skills_manager") as mock_get_mgr:
                
                mock_mgr = mock.Mock()
                mock_mgr.get_skill.return_value = {"code": "print('hello')"}
                mock_get_mgr.return_value = mock_mgr
                
                result = agent.read_skill("test_skill")
                assert result == "print('hello')"

    async def test_refresh_scratchpad_success(self):
        """Test successful scratchpad refresh."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            with mock.patch("graph_rlm.backend.src.core.agent.scratchpad_builder.build_scratchpad", 
                           return_value="Current scratchpad content"):
                result = await agent._refresh_scratchpad(
                    session_id="session-1",
                    root_session_id="root-1",
                    task="test task",
                    current_step=1,
                    max_steps=10,
                    current_round_id="round-1"
                )
                assert result == "Current scratchpad content"
                agent.emit_event.assert_called_with("scratchpad_text", content=result, is_internal=True)

    async def test_sync_thimac_success(self):
        """Test successful Thimac synchronization."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator") as mock_nav, \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory") as mock_memory:
            
            mock_nav_inst = mock.Mock()
            mock_nav_inst.compute_compression_progress.return_value = 0.5
            mock_nav.return_value = mock_nav_inst
            
            mock_mem_inst = mock.Mock()
            mock_event = mock.Mock()
            mock_event.to_dict.return_value = {"id": "thought-1"}
            mock_event.operation_reason = "Test"
            mock_mem_inst.ingest_thought.return_value = mock_event
            mock_memory.return_value = mock_mem_inst
            
            agent = Agent()
            
            with mock.patch("graph_rlm.backend.src.core.agent.summarize_event", return_value="gist"):
                result = await agent._sync_thimac(
                    thought_id="thought-1",
                    prompt="test prompt",
                    status="success",
                    result="test result",
                    step=1,
                    session_id="session-1",
                    round_id="round-1"
                )
                assert result["id"] == "thought-1"
                mock_mem_inst.ingest_thought.assert_called()
                agent.db.create_thought_node.assert_called()

    async def test_flush_memory_chain_success(self):
        """Test successful memory chain flushing to DB."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            
            mock_event1 = mock.Mock()
            mock_event1.thought_id = "thought-1"
            mock_event1.parent_id = None
            mock_event1.session_id = "session-1"
            mock_event1.root_session_id = "session-1"
            
            mock_event2 = mock.Mock()
            mock_event2.thought_id = "thought-2"
            mock_event2.parent_id = "thought-1"
            mock_event2.session_id = "session-1"
            mock_event2.root_session_id = "session-1"
            
            agent.morph_memory.all_events = [mock_event1, mock_event2]
            
            await agent._flush_memory_chain("thought-2")
            assert agent.db.create_thought_node.call_count == 2

    async def test_create_system_node_success(self):
        """Test successful system node creation."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            
            with mock.patch.object(agent, "_sync_thimac", return_value={"id": "sys-1"}) as mock_sync:
                node_id = await agent.create_system_node(
                    logical_id="SYS-TASK",
                    summary="System summary",
                    session_id="session-1"
                )
                assert node_id is not None
                mock_sync.assert_called()

    async def test_emit_event_success(self):
        """Test successful event emission via ContextVar."""
        from graph_rlm.backend.src.core.state import execution_events
        import queue
        
        q = queue.Queue()
        token = execution_events.set(q)
        try:
            with mock.patch("graph_rlm.backend.src.core.agent.db"), \
                 mock.patch("graph_rlm.backend.src.core.agent.llm"), \
                 mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
                 mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
                 mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
                
                agent = Agent()
                agent.emit_event("thinking", content="Test thinking")
                
                event = q.get_nowait()
                assert event["type"] == "thinking"
                assert event["content"] == "Test thinking"
        finally:
            execution_events.reset(token)

    async def test_initialize_turn(self):
        """Test turn initialization."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime") as mock_runtime, \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator") as mock_nav, \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"), \
             mock.patch("graph_rlm.backend.src.core.agent.agent_state") as mock_state:
            
            mock_runtime.return_value.execute = mock.AsyncMock(return_value="executed")
            mock_nav.return_value.compute_compression_progress.return_value = 0.0
            
            from graph_rlm.backend.src.core.state import ExecutionState
            state = ExecutionState(turn_id=5)
            mock_state.get.return_value = state
            
            agent = Agent()
            agent.emit_event = mock.Mock()
            
            with mock.patch("graph_rlm.backend.src.core.agent.scratchpad_builder.build_scratchpad", 
                           return_value="pad"), \
                 mock.patch("graph_rlm.backend.src.core.agent.summarize_event", return_value="gist"), \
                 mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", return_value="query"):
                
                await agent._initialize_turn(
                    prompt="test",
                    parent_id=None,
                    session_id="session-1",
                    depth=0,
                    root_session_id=None,
                    turn_id=5
                )
                assert agent.current_turn == 5
                assert agent.session_cache["root_session_id"] == "session-1"

    async def test_get_dashboard_metrics(self):
        """Test fetching dashboard metrics."""
        agent = Agent()
        agent.morph_memory = mock.Mock()
        mock_event = mock.Mock()
        mock_event.sheaf_score = 0.85
        mock_event.h0_rank = 2
        mock_event.omcd_score = 0.45
        mock_event.operation.value = "test_op"
        mock_event.metabolic_state = "ACTIVE"
        agent.morph_memory.all_events = [mock_event]
        
        agent.db = mock.Mock()
        mock_reflexion = mock.Mock()
        mock_reflexion.get.side_effect = lambda x: {"bar": "|||", "at_score": 0.5, "critique": "test"}.get(x)
        agent.db.query.return_value = [mock_reflexion]
        
        exec_state = mock.Mock()
        exec_state.branching_state = "STABLE"
        
        metrics = await agent._get_dashboard_metrics(exec_state, "session-1")
        assert metrics["sheaf_energy"] == "0.85"
        assert metrics["h0_rank"] == 2
        assert metrics["omcd_score"] == "0.45"
        assert metrics["thimac_op"] == "test_op"
        assert metrics["slac_meter"] == "|||"

    async def test_initialize_step_basic(self):
        """Test basic step initialization."""
        agent = Agent()
        agent.skills_manager = mock.Mock()
        agent.morph_memory = mock.Mock()
        agent.morph_memory.get_gestalt_string.return_value = "gestalt"
        
        turn_ctx = {
            "task_profile": {"persona": "test", "role": "execution"},
            "exec_state": mock.Mock(),
            "relevant_axioms": []
        }
        
        with mock.patch("graph_rlm.backend.src.core.agent.build_system_prompt", return_value="system"), \
             mock.patch("graph_rlm.backend.src.core.agent.meta_agents") as mock_meta:
            
            mock_meta.evaluate_coherence.return_value = False
            agent._get_dashboard_metrics = mock.AsyncMock(return_value={"sheaf_energy": "0.1"})
            
            await agent._initialize_step(1, "session-1", turn_ctx)
            
            assert "system_prompt" in turn_ctx
            assert turn_ctx["morph_gestalt"] == "gestalt"
            assert turn_ctx["logical_id"] == "session-:T1:S1"

    async def test_initialize_step_with_soar_and_energy(self):
        """Test step initialization with SOAR guidance and high energy."""
        agent = Agent()
        agent.skills_manager = mock.Mock()
        agent.morph_memory = mock.Mock()
        agent.morph_memory.get_gestalt_string.return_value = "gestalt"
        agent.emit_event = mock.Mock()
        
        turn_ctx = {
            "task_profile": {"persona": "test", "role": "execution"},
            "exec_state": mock.Mock(),
            "relevant_axioms": [],
            "task": "test goal"
        }
        
        with mock.patch("graph_rlm.backend.src.core.agent.build_system_prompt", return_value="system"), \
             mock.patch("graph_rlm.backend.src.core.agent.meta_agents") as mock_meta:
            
            mock_meta.evaluate_coherence.return_value = False
            # SOAR result
            mock_meta.run_cognitive_cycle = mock.AsyncMock(return_value={
                "phase": "APPLICATION",
                "operator": {
                    "action": "test action",
                    "tool": "test tool",
                    "rationale": "test rationale",
                    "preference": "1.0"
                }
            })
            
            agent._get_dashboard_metrics = mock.AsyncMock(return_value={
                "sheaf_energy": "0.9",
                "slac_critique": "mismatch"
            })
            
            await agent._initialize_step(1, "session-1", turn_ctx)
            
            # Check if high energy triggered resolution warning
            assert "TOPOLOGICAL DEFECT DETECTED" in turn_ctx["system_prompt"]
            # Check if SOAR guidance was added (step 1 % 3 == 1)
            assert "SOAR COGNITIVE GUIDANCE" in turn_ctx["system_prompt"]
            # Check events
            agent.emit_event.assert_any_call(
                "TOPOLOGICAL_RESOLUTION_REQUIRED",
                content=mock.ANY,
                tag="SYSTEM"
            )
            agent.emit_event.assert_any_call(
                "SOAR_APPLICATION",
                content=mock.ANY,
                tag="SOAR"
            )

    async def test_generate_thought_molhit_phases(self):
        """Test thought generation with MolHIT phases."""
        agent = Agent()
        agent.runtime = mock.Mock()
        agent.runtime.stopping = False
        agent.llm = mock.Mock()
        agent.llm.config = {"provider": "anthropic"}
        agent.emit_event = mock.Mock()
        
        with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate") as mock_gen, \
             mock.patch("graph_rlm.backend.src.core.agent.intelli_synth") as mock_synth:
            
            # Mock two phases of generation
            mock_gen.side_effect = ["INTENT: test\nTOOLS: rlm", "```python\nprint('hi')\n```"]
            mock_synth.introspective_probe = mock.AsyncMock(return_value=None)
            
            thought = await agent._generate_thought("system", "context", "session-1", 1)
            
            assert "print('hi')" in thought
            assert mock_gen.call_count == 2
            agent.emit_event.assert_any_call("debug_thought", content="[MolHIT] Phase 1: Topological Diffusion...")
            agent.emit_event.assert_any_call("debug_thought", content="[MolHIT] Phase 3: Node Instantiation (Code Generation)...")

    async def test_generate_thought_with_healing(self):
        """Test thought generation with introspective healing."""
        agent = Agent()
        agent.runtime = mock.Mock()
        agent.runtime.stopping = False
        agent.llm = mock.Mock()
        agent.llm.config = {"provider": "anthropic"}
        agent.emit_event = mock.Mock()
        agent.create_system_node = mock.AsyncMock()
        
        with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate") as mock_gen, \
             mock.patch("graph_rlm.backend.src.core.agent.intelli_synth") as mock_synth:
            
            # 1st attempt: Phase 1, Phase 3 -> Healing triggered
            # 2nd attempt: Phase 1, Phase 3 -> Success
            mock_gen.side_effect = [
                "INTENT: fail", "bad code", 
                "INTENT: success", "good code"
            ]
            mock_synth.introspective_probe = mock.AsyncMock(side_effect=[
                {"type": "SYNTAX_ERROR", "message": "fix it", "hint": "use quotes"},
                None
            ])
            
            thought = await agent._generate_thought("system", "context", "session-1", 1)
            
            assert thought == "good code"
            assert mock_gen.call_count == 4
            agent.emit_event.assert_any_call("thinking", content=mock.ANY, tag="SYSTEM")

    async def test_process_response_intentions(self):
        """Test processing response for code and intentions."""
        agent = Agent()
        response = """
        <distal_intention>Explore deep</distal_intention>
        <proximal_intention>Run tool</proximal_intention>
        <motor_intention>Click button</motor_intention>
        ```python
        print('hello')
        ```
        """
        rlm_ctx = mock.Mock()
        code, intent = await agent._process_response(response, rlm_ctx)
        
        assert "print('hello')" in code
        from graph_rlm.backend.src.core.agent import ThimacIntention
        assert intent == ThimacIntention.MOTOR
        assert rlm_ctx.proximal_intention == "Click button"

    async def test_execute_action_success(self):
        """Test successful action execution."""
        agent = Agent()
        agent.runtime = mock.Mock()
        agent.runtime.stopping = False
        agent.execution_logs = {"session-1": ["tool_a"]}
        
        agent._execute_code = mock.AsyncMock(return_value=("output", False, [], "hash"))
        agent._check_verification = mock.Mock()
        
        output, failed, tools, c_hash = await agent._execute_action(
            "print('hi')", "thought-1", "session-1", "root-1", "prompt", 0, 1
        )
        
        assert output == "output"
        assert not failed
        assert tools == ["tool_a"]
        assert c_hash == "hash"
        assert agent.execution_logs["session-1"] == []

    async def test_validate_and_finalize_success(self):
        """Test validation and finalization success path."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        agent.morph_memory = mock.Mock()
        agent.morph_memory.all_events = []
        agent.db = mock.Mock()
        agent.active_repls = {"session-1": "repl-1"}
        agent.session_cache = {"task_embedding": [0.1]}
        
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state") as mock_state, \
             mock.patch("graph_rlm.backend.src.core.agent.sheaf") as mock_sheaf, \
             mock.patch("graph_rlm.backend.src.core.agent.dreamer") as mock_dreamer:
            
            agent._verify_epistemic_integrity = mock.Mock(return_value={"status": "PASS"})
            mock_sheaf.check_axiomatic_consistency = mock.AsyncMock(return_value={"status": "AXIOMATIC_OK"})
            mock_dreamer.validate_response = mock.AsyncMock(return_value={"status": "valid", "message": "Good"})
            mock_dreamer.dream_cycle = mock.AsyncMock()
            agent._generate_validated_response = mock.AsyncMock(return_value="final answer")
            agent.create_system_node = mock.AsyncMock(return_value="thought-final")
            agent._flush_memory_chain = mock.AsyncMock()
            
            result = await agent._validate_and_finalize(
                "RLM_FINAL_OUTPUT: results", "context", "prompt", "session-1", "root-1", 1, "round-1", "repl-1", False
            )
            
            assert result is True
            assert agent.final_result == "final answer"
            agent.emit_event.assert_any_call("RLM_FINAL_OUTPUT", content="final answer")

    async def test_validate_and_finalize_epistemic_retry(self):
        """Test validation failure due to epistemic integrity."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        agent.create_system_node = mock.AsyncMock(return_value="thought-warning")
        
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state"):
            agent._verify_epistemic_integrity = mock.Mock(return_value={
                "status": "RETRY", 
                "flags": ["HALLUCINATION"]
            })
            
            result = await agent._validate_and_finalize(
                "RLM_FINAL_OUTPUT: results", "context", "prompt", "session-1", "root-1", 1, "round-1", "repl-1", False
            )
            
            assert result is False
            assert agent.final_result is None
            agent.emit_event.assert_any_call("system_event", content=mock.ANY, tag="REFLEXION")

    async def test_validate_and_finalize_axiom_violation(self):
        """Test validation failure due to axiomatic violation."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        agent.create_system_node = mock.AsyncMock(return_value="thought-axiom")
        
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state"), \
             mock.patch("graph_rlm.backend.src.core.agent.sheaf") as mock_sheaf:
            
            agent._verify_epistemic_integrity = mock.Mock(return_value={"status": "PASS"})
            mock_sheaf.check_axiomatic_consistency = mock.AsyncMock(return_value={
                "status": "AXIOMATIC_VIOLATION",
                "critique": "dangerous code"
            })
            
            result = await agent._validate_and_finalize(
                "RLM_FINAL_OUTPUT: results", "context", "prompt", "session-1", "root-1", 1, "round-1", "repl-1", False
            )
            
            assert result is False
            assert agent.final_result is None
            agent.emit_event.assert_any_call("system_event", content=mock.ANY, tag="SHEAF")

    async def test_validate_and_finalize_dreamer_rejected(self):
        """Test validation failure due to dreamer rejection."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        agent.create_system_node = mock.AsyncMock()
        
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state"), \
             mock.patch("graph_rlm.backend.src.core.agent.dreamer") as mock_dreamer:
            
            agent._verify_epistemic_integrity = mock.Mock(return_value={"status": "PASS"})
            mock_dreamer.validate_response = mock.AsyncMock(return_value={
                "status": "rejected",
                "instruction": "try again",
                "reasons": ["vague"]
            })
            
            result = await agent._validate_and_finalize(
                "RLM_FINAL_OUTPUT: results", "context", "prompt", "session-1", "root-1", 1, "round-1", "repl-1", False
            )
            
            assert result is False
            assert agent.last_dream_insight == "try again"

    async def test_validate_and_finalize_budget_exhausted(self):
        """Test validation path when dreamer budget is exhausted."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state") as mock_state, \
             mock.patch("graph_rlm.backend.src.core.agent.dreamer") as mock_dreamer:
            
            # Properly mock the state attributes to avoid MagicMock concatenation issues
            mock_state_obj = mock.Mock()
            mock_state_obj.final_result = None
            mock_state.get.return_value = mock_state_obj
            
            agent._verify_epistemic_integrity = mock.Mock(return_value={"status": "PASS"})
            mock_dreamer.validate_response = mock.AsyncMock(return_value={
                "status": "exhausted",
                "instruction": "too expensive"
            })
            
            result = await agent._validate_and_finalize(
                "RLM_FINAL_OUTPUT: results", "context", "prompt", "session-1", "root-1", 1, "round-1", "repl-1", False
            )
            assert result is True
            assert "force-accepted" in agent.final_result
            assert "too expensive" in agent.final_result

    def test_emit_terminal_report(self):
        """Test consolidated terminal reporting."""
        agent = Agent()
        agent.emit_event = mock.Mock()
        agent.final_result = "final answer"
        
        agent._emit_terminal_report("TEST_REASON", "test details")
        
        assert agent._final_output_emitted is True
        agent.emit_event.assert_called_once()
        args, kwargs = agent.emit_event.call_args
        assert args[0] == "RLM_FINAL_OUTPUT"
        assert "TEST_REASON" in kwargs["content"]
        assert "final answer" in kwargs["content"]

    def test_check_verification(self):
        """Test Rule 5 verification detection."""
        agent = Agent()
        with mock.patch("graph_rlm.backend.src.core.agent.agent_state") as mock_state:
            state = mock.Mock()
            state.pending_side_effects = ["file_write"]
            mock_state.get.return_value = state
            
            # No verification
            agent._check_verification("print('hi')", [])
            assert len(state.pending_side_effects) == 1
            
            # Code verification
            agent._check_verification("os.path.exists('file')", [])
            assert len(state.pending_side_effects) == 0
            
            # Tool verification
            state.pending_side_effects = ["file_write"]
            agent._check_verification("print('hi')", ["view_file('test.py')"])
            assert len(state.pending_side_effects) == 0

    def test_extract_code(self):
        """Test python code extraction."""
        agent = Agent()
        with mock.patch("graph_rlm.backend.src.core.guardrails.extract_python_code") as mock_extract:
            mock_extract.return_value = "print('hello')"
            result = agent._extract_code("```python\nprint('hello')\n```")
            assert result == "print('hello')"
            mock_extract.assert_called_once()

    async def test_run_recursive_logic_success(self):
        """Test core recursive logic loop - success path."""
        agent = Agent()
        agent.stop_requested = False
        agent.final_result = "done"
        agent._final_output_emitted = False
        
        # Mocks
        agent._initialize_turn = mock.AsyncMock(return_value={
            "step": 0, "max_steps": 5, "system_prompt": "sys", "pad": "pad", 
            "root_id": "root", "round_id": "round", "repl_id": "repl"
        })
        agent._initialize_step = mock.AsyncMock()
        agent._generate_thought = mock.AsyncMock(return_value="thought")
        agent.llm = mock.AsyncMock()
        agent.llm.get_embedding = mock.AsyncMock(return_value=[0.1])
        agent._sync_thimac = mock.AsyncMock()
        agent._refresh_scratchpad = mock.AsyncMock(return_value="new_pad")
        agent._process_response = mock.AsyncMock(return_value=("code", {}))
        agent._execute_action = mock.AsyncMock(return_value=("summary", False, [], "hash"))
        
        async def mock_validate(*args, **kwargs):
            agent._final_output_emitted = True
            return True
            
        agent._validate_and_finalize = mock.AsyncMock(side_effect=mock_validate)
        agent._emit_terminal_report = mock.Mock()
        
        with mock.patch("graph_rlm.backend.src.core.repe.repe") as mock_repe:
            mock_repe.scan_thought.return_value = {"scores": {}, "rationale": "ok"}
            
            result = await agent.query_sync("prompt", "parent", "session", 1, "root")
            
            assert result == "done"
            agent._initialize_turn.assert_called_once()
            agent._generate_thought.assert_called_once()
            agent._validate_and_finalize.assert_called_once()
            # Should break loop and NOT emit terminal report because validation passed
            agent._emit_terminal_report.assert_not_called()

    async def test_run_recursive_logic_max_steps(self):
        """Test core recursive logic loop - max steps reached."""
        agent = Agent()
        agent.stop_requested = False
        
        # Mocks
        agent._initialize_turn = mock.AsyncMock(return_value={
            "step": 0, "max_steps": 1, "system_prompt": "sys", "pad": "pad", 
            "root_id": "root", "round_id": "round", "repl_id": "repl"
        })
        agent._initialize_step = mock.AsyncMock()
        agent._generate_thought = mock.AsyncMock(return_value="thought")
        agent.llm = mock.AsyncMock()
        agent.llm.get_embedding = mock.AsyncMock(return_value=None)
        agent._sync_thimac = mock.AsyncMock()
        agent._refresh_scratchpad = mock.AsyncMock(return_value="new_pad")
        agent._process_response = mock.AsyncMock(return_value=(None, {}))
        agent._validate_and_finalize = mock.AsyncMock(return_value=False)
        agent._emit_terminal_report = mock.Mock()
        
        result = await agent.query_sync("prompt", "parent", "session", 1, "root")
        
        assert "stopped" in result
        agent._emit_terminal_report.assert_called_once_with("MAX_STEPS_REACHED", mock.ANY)

    async def test_stream_query_basic(self):
        """Test stream_query basic loop."""
        with mock.patch("graph_rlm.backend.src.core.agent.db"), \
             mock.patch("graph_rlm.backend.src.core.agent.llm"), \
             mock.patch("graph_rlm.backend.src.core.agent.AgentRuntime"), \
             mock.patch("graph_rlm.backend.src.core.agent.Navigator"), \
             mock.patch("graph_rlm.backend.src.core.agent.ThimacMemory"):
            
            agent = Agent()
            
            # Mock query_sync to do nothing
            async def mock_query_sync(*args, **kwargs):
                agent.emit_event("thinking", content="done")
                agent.final_result = "Final Answer"
            
            with mock.patch.object(agent, "query_sync", side_effect=mock_query_sync):
                events = []
                async for event in agent.stream_query("test prompt"):
                    events.append(event)
                
                assert len(events) >= 2
                # RLM_INITIAL_RESPONSE because query_sync sets final_result
                types = [e["type"] for e in events]
                assert "thinking" in types
                assert "RLM_INITIAL_RESPONSE" in types

    async def test_execute_code_success(self):
        """Test successful code execution in subprocess."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        agent.runtime.execute = mock.AsyncMock(return_value=("output", "", "result", 0))
        agent.active_repls = {"session-1": "repl-1"}
        
        output, failed, summary, c_hash = await agent._execute_code(
            "print('hi')", "thought-1", "session-1", "root-1", "input", 1, 1
        )
        
        assert "output" in output
        assert "Return Value: result" in output
        assert failed is False
        assert "output Return Value: result" in summary
        assert c_hash is not None

    async def test_execute_code_retry_missing_module(self):
        """Test code execution retry when ModuleNotFoundError occurs."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        agent.emit_event = mock.Mock()
        agent.install_package = mock.AsyncMock(return_value="Successfully installed pkg")
        
        # 1st call fails with ModuleNotFoundError, 2nd call succeeds
        agent.runtime.execute.side_effect = [
            ("", "Traceback: ModuleNotFoundError: No module named 'pkg'", None, 1),
            ("success", "", "done", 0)
        ]
        
        output, failed, summary, _ = await agent._execute_code(
            "import pkg", "thought-1", "session-1"
        )
        
        assert failed is False
        assert "success" in output
        assert agent.install_package.call_count == 1
        agent.emit_event.assert_any_call("thinking", content=mock.ANY)

    async def test_execute_code_truncation(self):
        """Test output truncation and log saving."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        large_output = "a" * 3000
        agent.runtime.execute = mock.AsyncMock(return_value=(large_output, "", None, 0))
        
        with mock.patch("graph_rlm.backend.src.core.agent.settings") as mock_settings, \
             mock.patch("pathlib.Path.mkdir"), \
             mock.patch("pathlib.Path.write_text"):
            
            mock_settings.KNOWLEDGE_BASE_PATH = "/tmp/kb"
            output, failed, summary, _ = await agent._execute_code(
                "print('big')", "thought-1", "session-1"
            )
            
            assert "Output truncated" in output
            assert len(summary) < 2000

    def test_stop_generation(self):
        """Test stopping the agent."""
        agent = Agent()
        agent.global_stop_event = mock.Mock()
        agent.stop_requested = False
        
        agent.stop_generation()
        
        assert agent.stop_requested is True
        agent.global_stop_event.set.assert_called_once()

    async def test_generate_validated_response(self):
        """Test final report synthesis."""
        agent = Agent()
        agent.db = mock.Mock()
        # Mocking node properties
        mock_node = mock.Mock()
        mock_node.properties = {"type": "thought", "content": "did X", "status": "success"}
        agent.db.query.return_value = [{"n": mock_node}]
        
        with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", 
                       return_value="# RLM_DREAMER_VALIDATED\nDone") as mock_gen:
            
            result = await agent._generate_validated_response("root-1", "task-1")
            
            assert "# RLM_DREAMER_VALIDATED" in result
            mock_gen.assert_called_once()
            args, kwargs = mock_gen.call_args
            assert "did X" in args[0]

    def test_verify_epistemic_integrity_lazy(self):
        """Test epistemic check flagging laziness."""
        agent = Agent()
        result = agent._verify_epistemic_integrity(
            thought_trace="too short",
            task_requirements="Analyze the whole system",
            execution_log=[]
        )
        assert result["status"] == "RETRY"
        assert any("LAZINESS" in f for f in result["flags"])

    def test_verify_epistemic_integrity_obsequious(self):
        """Test epistemic check flagging obsequiousness."""
        agent = Agent()
        result = agent._verify_epistemic_integrity(
            thought_trace="You are absolutely right about everything.",
            task_requirements="task",
            execution_log=["something"]
        )
        assert any("OBSEQUIOUSNESS" in f for f in result["flags"])

    def test_verify_epistemic_integrity_reward_hacking(self):
        """Test epistemic check flagging reward hacking."""
        agent = Agent()
        # Claims completion with TODO
        result = agent._verify_epistemic_integrity(
            thought_trace="I am final answer [TODO] fix later",
            task_requirements="task",
            execution_log=[]
        )
        assert any("TEMPLATE_HALLUCINATION" in f for f in result["flags"])

    async def test_detect_required_axioms_agentic(self):
        """Test agentic axiom discovery."""
        agent = Agent()
        agent.skills_manager = mock.AsyncMock()
        agent.skills_manager.find_similar_skills.return_value = [
            {"name": "phys_safe", "score": 0.9, "tags": ["physics"]}
        ]
        
        with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", 
                       return_value="physics safety rules") as mock_gen:
            
            axioms = await agent._detect_required_axioms_agentic("move ball", "ball.x += 1")
            
            assert "physics" in axioms
            mock_gen.assert_called_once()

    async def test_execute_code_detect_runtime_error(self):
        """Test detection of RuntimeError in output even if exit_code is 0."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        # Exit code 0 but has Traceback in output
        agent.runtime.execute = mock.AsyncMock(return_value=("Traceback (most recent call last): error", "", None, 0))
        
        output, failed, summary, _ = await agent._execute_code("code", "t1", "s1")
        assert failed is True

    async def test_execute_code_io_error_on_log(self):
        """Test IO error when saving large output log."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        agent.runtime.execute = mock.AsyncMock(return_value=("a" * 3000, "", None, 0))
        
        with mock.patch("graph_rlm.backend.src.core.agent.settings") as mock_settings, \
             mock.patch("pathlib.Path.mkdir", side_effect=OSError("Disk Full")):
            mock_settings.KNOWLEDGE_BASE_PATH = "/tmp"
            output, failed, summary, _ = await agent._execute_code("code", "t1", "s1")
            assert "EMERGENCY TRUNCATION" in output
            assert summary == "Output Truncated (Log Error)"

    async def test_execute_code_empty_output(self):
        """Test handling of empty execution output."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        agent.runtime.execute = mock.AsyncMock(return_value=("  ", "", None, 0))
        
        output, failed, summary, _ = await agent._execute_code("code", "t1", "s1")
        assert "No output captured" in output
        assert summary == "Success (No Output)"

    def test_verify_epistemic_integrity_complex_reward_hacking(self):
        """Test complex reward hacking (code but no tool interaction)."""
        agent = Agent()
        # Complex task, claims final answer with code, but execution_log is empty
        result = agent._verify_epistemic_integrity(
            thought_trace="final answer ```python\nprint('hack')\n```",
            task_requirements="Analyze the data",
            execution_log=[]
        )
        assert any("REWARD_HACKING" in f for f in result["flags"])

    def test_verify_epistemic_integrity_failed_artifacts(self):
        """Test reward hacking when logs show failure."""
        agent = Agent()
        result = agent._verify_epistemic_integrity(
            thought_trace="final answer",
            task_requirements="task",
            execution_log=["Error: failed to find file"]
        )
        assert any("REWARD_HACKING" in f for f in result["flags"])

    def test_verify_epistemic_integrity_report_laziness(self):
        """Test laziness check for short reports."""
        agent = Agent()
        result = agent._verify_epistemic_integrity(
            thought_trace="final answer",
            task_requirements="Write a full report",
            execution_log=["short line"]
        )
        assert any("LAZINESS" in f for f in result["flags"])

    async def test_detect_required_axioms_agentic_failure(self):
        """Test fallback in axiom discovery on error."""
        agent = Agent()
        with mock.patch("graph_rlm.backend.src.core.agent.protected_llm_generate", 
                       side_effect=ValueError("LLM Error")):
            axioms = await agent._detect_required_axioms_agentic("prompt", "code")
            assert axioms == ["general"]

    async def test_get_dashboard_metrics_failure(self):
        """Test metrics retrieval failure."""
        agent = Agent()
        agent.db = mock.Mock()
        agent.db.query.return_value = []
        agent.morph_memory = mock.MagicMock()
        # Trigger AttributeError on index access [-1]
        agent.morph_memory.all_events.__getitem__.side_effect = AttributeError("fail")
        
        metrics = await agent._get_dashboard_metrics(mock.Mock(), "s1")
        assert metrics == {}

    async def test_generate_thought_stop_requested(self):
        """Test thought generation when stop is requested."""
        agent = Agent()
        agent.stop_requested = True
        
        res = await agent._generate_thought("prompt", "ctx", "s1", 1)
        assert res == ""

    async def test_query_sync_reflexion(self):
        """Test reflexion self-healing in query_sync."""
        agent = Agent()
        agent.db = mock.Mock()
        agent.llm = mock.AsyncMock()
        agent.llm.get_embedding.return_value = [0.1]
        agent._sync_thimac = mock.AsyncMock()
        agent._refresh_scratchpad = mock.AsyncMock(return_value="pad")
        agent._process_response = mock.AsyncMock(return_value=("code", "plan"))
        
        # Mock execution failure
        agent._execute_action = mock.AsyncMock(return_value=("error", True, [], "hash"))
        
        # Patch the source module
        with mock.patch("graph_rlm.backend.src.core.dream.Dreamer") as mock_dreamer_class:
            mock_dreamer = mock.Mock()
            mock_dreamer.dream_cycle = mock.AsyncMock(return_value={"insight": "Fix it"})
            mock_dreamer_class.return_value = mock_dreamer
            
            # Mock _generate_thought to return something then None to break loop
            agent._generate_thought = mock.AsyncMock()
            agent._generate_thought.side_effect = ["thought", None]
            
            # Set required state for query_sync
            agent.stop_requested = False
            
            # Mock initialize_turn to return a valid context
            agent._initialize_turn = mock.AsyncMock(return_value={
                "step": 0, "max_steps": 10, "root_id": "r1", "round_id": "rd1", 
                "repl_id": "repl1", "pad": "pad", "system_prompt": "prompt",
                "task_profile": {"persona": "Test"},
                "exec_state": mock.Mock(),
                "prompt": "prompt"
            })
            agent._initialize_step = mock.AsyncMock()
            
            await agent.query_sync("task", session_id="s1")
            
    async def test_stop_generation_with_event(self):
        """Test stop_generation with global_stop_event."""
        agent = Agent()
        agent.global_stop_event = mock.Mock()
        agent.stop_generation()
        agent.global_stop_event.set.assert_called_once()
        assert agent.stop_requested is True

    @mock.patch("graph_rlm.backend.src.core.agent.RLMInterface")
    async def test_execute_action_rlm_interface_failure(self, mock_rlm):
        """Test _execute_action fallback when RLMInterface fails."""
        agent = Agent()
        agent.runtime = mock.AsyncMock()
        agent.runtime.execute = mock.AsyncMock(return_value=("out", "err", "res", 0))
        mock_rlm.side_effect = RuntimeError("Failed init")
        
        # _execute_action(code, thought_id, session_id, root_id, prompt, turn_id, step)
        res = await agent._execute_action("code", "t1", "s1", "r1", "p1", 1, 0)
        # Should complete despite RLMInterface failure
        assert res[1] is False # execution_failed

    async def test_query_sync_fragmentation(self):
        """Test fragmentation detection (H0 > 1) in query_sync."""
        agent = Agent()
        agent.db = mock.Mock()
        agent.llm = mock.AsyncMock()
        agent.llm.get_embedding.return_value = [0.1]
        agent._sync_thimac = mock.AsyncMock()
        agent.current_thought_id = "parent_id"
        
        # exec_state with H0 > 1
        mock_state = mock.Mock()
        mock_state.last_h0_rank = 2
        mock_state.stop_requested = False
        
        agent._initialize_turn = mock.AsyncMock(return_value={
            "step": 0, "max_steps": 1, "root_id": "r1", "round_id": "rd1", 
            "repl_id": "repl1", "pad": "pad", "system_prompt": "prompt",
            "task_profile": {"persona": "Test"},
            "exec_state": mock_state,
            "prompt": "prompt",
            "step": 0
        })
        agent._initialize_step = mock.AsyncMock()
        agent._generate_thought = mock.AsyncMock(return_value="thought")
        agent._process_response = mock.AsyncMock(return_value=("code", "plan"))
        agent._execute_action = mock.AsyncMock(return_value=("out", False, [], "hash"))
        agent._refresh_scratchpad = mock.AsyncMock()
        agent.create_system_node = mock.AsyncMock()
        agent._should_stop_query = mock.AsyncMock(return_value=False)
        agent.llm.get_embedding = mock.AsyncMock(return_value=[0.1] * 3072)
        
        from graph_rlm.backend.src.core.state import agent_state
        token = agent_state.set(mock_state)
        agent.stop_requested = False
        try:
            # Patch the module where repe is imported from
            with mock.patch("graph_rlm.backend.src.core.repe.repe") as mock_repe:
                mock_repe.scan_thought.return_value = {"scores": {}}
                await agent.query_sync("task", "s1")
        finally:
            agent_state.reset(token)
        
        agent._initialize_turn.assert_called()
        agent.create_system_node.assert_called()

    async def test_generate_validated_response_empty_trace(self):
        """Test _generate_validated_response with empty trace."""
        agent = Agent()
        agent.db = mock.Mock()
        agent.db.query.return_value = [] # No nodes
        agent.final_result = "Final Result"
        
        # This function returns early if trace is empty
        res = await agent._generate_validated_response("r1", "task")
        assert "Final Result" in res
        assert "# RLM_DREAMER_VALIDATED" in res

    @mock.patch("graph_rlm.backend.src.core.services.circuit.protected_llm_generate", new_callable=mock.AsyncMock)
    async def test_generate_validated_response_exception(self, mock_gen):
        """Test _generate_validated_response exception handling."""
        agent = Agent()
        agent.db = mock.Mock()
        agent.db.query.side_effect = ValueError("Query error")
        
        res = await agent._generate_validated_response("r1", "task")
        assert "Error generating validation" in res

    async def test_initialize_turn_breaker_injection(self):
        """Test BREAKER protocol injection in _initialize_turn."""
        agent = Agent()
        agent._sync_thimac = mock.AsyncMock()
        agent._generate_task_profile = mock.AsyncMock(return_value={})
        agent.db = mock.Mock()
        
        with mock.patch("graph_rlm.backend.src.core.agent.meta_agents") as mock_meta:
            # Configure meta_agents to trigger breaker
            mock_meta.should_spawn_breakers.return_value = True
            mock_meta.get_breaker_instructions.return_value = "BREAKER_INST"
            
            res = await agent._initialize_turn("complex task", None, "s1", 0, None, 1)
            
            assert "BREAKER_INST" in res["prompt"]
            mock_meta.get_breaker_instructions.assert_called()

    async def test_initialize_turn_failure_fallback(self):
        """Test fallback when turn initialization fails in _initialize_turn."""
        agent = Agent()
        # Mock _sync_thimac to raise error
        agent._sync_thimac = mock.AsyncMock(side_effect=RuntimeError("Sync Failed"))
        
        res = await agent._initialize_turn("task", None, "s1", 0, None, 1)
        
        # Should have reached fallback and returned a valid dict
        assert "task_id" in res
        assert res["task_profile"]["persona"] == "Autonomous Generalist"

    async def test_process_response_exception(self):
        """Test _process_response exception handling."""
        agent = Agent()
        agent.llm = mock.AsyncMock()
        agent.llm.extract_json = mock.Mock(side_effect=Exception("Parse Error"))
        
        # Args: response_text, rlm_ctx
        code, intent = await agent._process_response("raw response", None)
        # _extract_code returns '' if no code found
        assert code == ""
        assert intent is None

    async def test_initialize_turn_missing_state(self):
        """Test _initialize_turn fallback when state is missing."""
        agent = Agent()
        from graph_rlm.backend.src.core.state import agent_state
        # Ensure state is None
        token = agent_state.set(None)
        try:
            # Args: prompt, parent_id, session_id, depth, root_session_id, turn_id
            # turn_id MUST be int
            res = await agent._initialize_turn("prompt", None, "s1", 0, "r1", 1)
            assert res["exec_state"] is not None
        finally:
            agent_state.reset(token)
