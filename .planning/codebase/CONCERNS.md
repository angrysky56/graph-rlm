# CONCERNS.md

## Technical Debt

### High Priority

#### 1. Test Isolation Complexity

**Issue**: Tests require careful dependency mocking BEFORE importing to prevent import-time errors.

**Evidence** (`tests/tests/test_agent_isolation.py`):

```python
# Mocking must happen BEFORE import
# to prevent import-time database connections
```

**Impact:**

- Easy to break tests by importing in wrong order
- Fragile test setup
- Debugging import cycles is difficult

**Mitigation:**

- Use `IsolatedAsyncioTestCase` patterns
- Document import order requirements
- Consider refactoring to dependency injection

#### 2. Axiom Version Management

**Issue**: Axiom purging and regeneration scripts exist but may not handle all edge cases.

**Evidence**:

- `scripts/purge_orphaned_axioms.py`
- `scripts/regenerate_tools.py`

**Impact:**

- Orphaned axioms may accumulate
- Regeneration may miss dependencies
- Axiom drift over time

**Mitigation:**

- Automated axiom garbage collection
- Version tracking for axioms
- Dependency graph for axioms

### Medium Priority

#### 3. Async Event Loop Management

**Issue**: `nest-asyncio` usage for REPL isolation may have edge cases.

**Evidence**:

- `nest-asyncio>=1.6.0` dependency

**Impact:**

- Potential event loop nesting bugs
- Hard to debug async state
- Race conditions possible

**Mitigation:**

- Comprehensive async tests
- Clear event loop lifecycle documentation
- Consider alternative isolation approaches

#### 4. MCP Server Configuration

**Issue**: `mcp_servers.json` configuration may become stale.

**Impact:**

- Server definitions may drift
- Missing required fields
- Hard to validate configurations

**Mitigation:**

- Schema validation for MCP config
- Version checking for servers
- Health checks for MCP connections

### Low Priority

#### 5. Token Counting Accuracy

**Issue**: `tiktoken` may not perfectly match all LLM tokenizers.

**Impact:**

- Cost estimation errors
- Context window limit confusion

**Mitigation:**

- Per-provider token counting
- Fallback to provider estimates

#### 6. Frontend-Backend Sync

**Issue**: React frontend state may drift from backend API.

**Impact:**

- UI errors on API changes
- Stale data displayed

**Mitigation:**

- TypeScript API client
- Versioned API endpoints

## Known Issues

### 1. Vector Index Rebuild

**Issue**: Vector indices may require manual rebuild after certain operations.

**Status**: Known, workaround in place
**Workaround**: `scripts/regenerate_tools.py`

### 2. Dreamer Consolidation

**Issue**: High-surprise event extraction may miss edge cases. Gatekeeper logic was previously too strict, causing runaway loops.

**Status**: Partial Fix (Gatekeeper relaxed)
**Workaround**: Manual review of `rules.md` updates. Runaway loop fixed in `agent.py`.

### 3. Graph Pruning

**Issue**: Automatic graph pruning may delete useful context.

**Status**: Known, conservative settings
**Workaround**: Manual graph inspection before pruning

## Security Concerns

### 1. API Key Exposure

**Concern**: Keys in environment variables need careful management.

**Mitigation:**

- Never commit `.env` files
- Use secrets management in production
- Rotate keys regularly

### 2. LLM Output Validation

**Concern**: Axiomatic verification may not catch all malicious outputs.

**Mitigation:**

- Additional output sanitization
- Rate limiting
- Human-in-the-loop for critical operations

### 3. MCP Tool Safety

**Concern**: Dynamically loaded MCP tools may have security implications.

**Mitigation:**

- Tool whitelisting
- Sandboxed execution
- Permission models

## Performance Concerns

### 1. Graph Query Latency

**Concern**: Large graphs may have slow query times.

**Mitigation:**

- Index optimization
- Query caching
- Graph partitioning

### 2. LLM API Latency

**Concern**: External LLM calls introduce latency.

**Mitigation:**

- Request batching
- Streaming responses
- Local caching

### 3. Axiom Verification Overhead

**Concern**: Running all axioms before each action may be slow.

**Mitigation:**

- Relevance-based axiom filtering
- Parallel axiom execution
- Axiom result caching

## Fragile Areas

### 1. Session State Management

**Area**: `graph_rlm/` session handling

**Why Fragile:**

- Complex state inheritance
- Multiple async contexts
- REPL isolation boundaries

**Protection:**

- Isolated test cases
- Session ID validation
- State reset mechanisms

### 2. Axiom Library Updates

**Area**: `scripts/purge_orphaned_axioms.py` and related

**Why Fragile:**

- Dependency tracking complex
- Cascade effects hard to predict
- Version compatibility

**Protection:**

- Version pinning
- Comprehensive test coverage
- Incremental updates

### 3. MCP Tool Loading

**Area**: `skills/` dynamic loading

**Why Fragile:**

- Runtime import errors
- Missing dependencies
- API changes

**Protection:**

- Strict validation
- Graceful degradation
- Version requirements

## Maintenance Issues

### Documentation Drift

**Issue**: Code comments and docs may not match implementation.

**Mitigation:**

- Automated documentation generation
- Code review for doc updates
- Linting for doc quality

### Test Maintenance

**Issue**: Many test files with complex setups.

**Mitigation:**

- Shared test fixtures
- Test pattern documentation
- Automated test organization

### Dependency Updates

**Issue**: Many external dependencies with varying update cycles.

**Mitigation:**

- Dependency pinning
- Automated vulnerability scanning
- Regular update cadence

## Areas Needing Attention

1. **Axiom Serialization**: Consider adding serialization format documentation
2. **Graph Query Optimization**: Profile and optimize slow queries
3. **Test Coverage Gaps**: Identify untested code paths
4. **Error Message Quality**: Improve error messages for debugging
5. **Logging Standardization**: Consistent logging across modules
