# Project Lessons Learned

_Capture post-mortems, failure modes, and prevention rules here based on the @code-process guidelines._

## Template

- **Failure Mode**: [What went wrong?]
- **Detection Signal**: [How did we find out? (Error, unexpected behavior, log output)]
- **Prevention Rule**: [What concrete action prevents this in the future?]
- **Date**: YYYY-MM-DD

---

## History

- **Failure Mode**: Agent hallucinated `FalkorDBProxy` class while executing autonomous sub-scripts in the REPL.
- **Detection Signal**: `kernel.py` suffered an `ImportError` on dynamically evaluated code.
- **Prevention Rule**: oMCD (metacognitive control) must remain integrated; a missing configuration file/metric display triggers extreme agent autonomy/hallucination to "fix" itself. Keep core metrics explicitly grounded in the TUI state string to prevent agent panic. Ensure constraint manifolds (energy conservation boundaries) restrict mathematically optimal but physically impossible code generation.
- **Date**: 2026-02-20
