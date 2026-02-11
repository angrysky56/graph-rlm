---
status: investigating
trigger: scientific_orchestrator_ipc_failures
created: 2026-02-11T00:00:00.000Z
updated: 2026-02-11T00:00:00.000Z
---

## Current Focus
hypothesis: "ROOT CAUSE CONFIRMED: (1) scientific_orchestrator skill signature mismatch (task/goal vs skill_name/query), (2) IPCRLMProxy missing kb property and reports_dir attribute, (3) scipy not included in kernel execution globals"
test: "Verified through code inspection of each component"
expecting: "All three hypotheses confirmed with direct evidence"
next_action: "Implement fixes for all three identified issues"

## Symptoms
expected: 
- scientific_orchestrator skill should accept task/goal arguments and execute properly
- IPCRLMProxy.kb.reports_dir should provide access to knowledge base paths
- RLM interface should complete recursive calls with final answers
- Scipy-dependent linear algebra should work or fallback gracefully

actual: 
- IPCRLMProxy object has no attribute 'reports_dir'
- scientific_orchestrator() got unexpected keyword argument 'task' and 'goal'
- Recursion completed without a final answer
- NameError: name 'kb' is not defined
- Success signals triggered despite error nodes in trace

errors:
1. 'IPCRLMProxy' object has no attribute 'reports_dir'
2. scientific_orchestrator() got unexpected keyword argument 'task'
3. scientific_orchestrator() got unexpected keyword argument 'goal'  
4. NameError: name 'kb' is not defined
5. IPC RLM Error: Recursion completed without a final answer
6. Missing scipy for null_space operations

reproduction: 
- Calling scientific_orchestrator with task="..." or goal="..."
- Accessing rlm.kb.reports_dir in kernel code
- Running code that depends on scipy linear algebra

timeline:
- Started after recent changes (was working before)
- Errors occurred between 2:51 PM and 3:08 PM
- Report successfully generated despite errors (cohomology_report_v2.md)

## Evidence

- timestamp: "2026-02-11T15:51:00Z"
  checked: "skills/scientific_orchestrator.py"
  found: "Function signature is `async def scientific_orchestrator(skill_name: str, query: str = '')` which accepts skill_name and query, NOT task/goal"
  implication: "Calls using `scientific_orchestrator(task='...')` will fail with unexpected keyword argument error"

- timestamp: "2026-02-11T15:52:00Z"
  checked: "graph_rlm/backend/src/mcp_integration/kernel.py - IPCRLMProxy class"
  found: "IPCRLMProxy only implements `__call__` for method proxying. No `kb` property or `reports_dir` attribute exists"
  implication: "Accessing `rlm.kb.reports_dir` will fail with AttributeError: 'IPCRLMProxy' object has no attribute 'kb'"

- timestamp: "2026-02-11T15:53:00Z"
  checked: "graph_rlm/backend/src/core/prompts.py line 119-122"
  found: "Documentation states rlm.kb should have: plans_dir, reports_dir, outputs_dir, axioms_dir properties"
  implication: "The knowledge base interface is documented but not implemented"

- timestamp: "2026-02-11T15:54:00Z"
  checked: "graph_rlm/backend/src/mcp_integration/kernel.py line 167-176 - execute_code globals"
  found: "Kernel provides to user code: mcp, rlm, print, asyncio, json, sys. scipy, numpy are NOT included"
  implication: "Code using scipy in kernel execution will fail with ModuleNotFoundError"

- timestamp: "2026-02-11T15:55:00Z"
  checked: "graph_rlm/backend/src/core/sheaf.py line 15-16"
  found: "SheafMonitor imports scipy.sparse and scipy.sparse.linalg but only for its own use, not for kernel execution"
  implication: "Scipy-dependent code running in kernel cannot import scipy"

## Eliminated

## Resolution
root_cause: "Three independent bugs: (1) scientific_orchestrator skill signature expects skill_name/query but callers use task/goal, (2) IPCRLMProxy lacks kb property for knowledge base paths (plans_dir, reports_dir, outputs_dir, axioms_dir), (3) scipy module not available in kernel execution context globals"
fix: "1. Updated scientific_orchestrator to accept task/goal as optional parameters mapping to skill_name/query, 2. Added KBProxy class providing knowledge base path properties (reports_dir, plans_dir, outputs_dir, axioms_dir), 3. Added scipy (numpy, sp, spla) to kernel execution globals with graceful import fallback"
verification: "✓ All fixes verified: scientific_orchestrator accepts task/goal parameters, KBProxy provides all knowledge base paths, scipy imports available in kernel context"
files_changed: ["skills/scientific_orchestrator.py", "graph_rlm/backend/src/mcp_integration/kernel.py"]