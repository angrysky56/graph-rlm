# Debug Diagnosis: Trace Failed, Suspect State

## Observations
1. **Trace Logs**: The request log `POST /api/v1/chat/completions` appears (200 OK), but **no debug prints** from inside `response_stream` appear in the terminal.
2. **Implication**: `StreamingResponse` is being returned, but the generator `response_stream` is **not being iterated**.
3. **Root Cause**: This typically happens when the client (Frontend) disconnects immediately after receiving the headers, or if an exception occurs *during* the creation of the response but *before* the generator starts (unlikely given the code structure).
4. **Model Saving Issue**: User mentioned "why isn't it saving my model?". This refers to the UI selection resetting on reload. `ModelSelector` just calls `onSelect`. It likely relies on parent state. If persistence is missing in `ChatInterface` or `Settings` context, it resets to default.

## Fix Plan
1. **Frontend Persistence**: ensuring `ChatInterface` or `SettingsProvider` saves the selected model to `localStorage` or backend config.
2. **Backend**: Verify `OPENROUTER_MODEL` in `.env` is updated when the user changes it (if that's the intended design), or if it's per-session. The endpoint `update_config` exists but might not be called by `ModelSelector`.
3. **Execution**:
    - Check `ChatInterface.tsx` to see how it handles model changes.
    - If `ModelSelector` only updates local state, we need to wire it to `api.updateConfig`.

## Why "Not Working"?
If the frontend is trying to talk to a model ID that doesn't exist (e.g. `x-ai/grok` if not supported by current provider config) and getting a silent error, or if the connection drops.
Wait, if the user picks 'Grok' but `LLM_PROVIDER` in backend is 'openrouter' (default), and `settings.get_llm_config()` returns the OpenRouter config...
The UI needs to send the *selected model* in the `ChatCompletionRequest`.
Let's check `api.ts` -> `streamChat`. It sends `...payload`. `ChatInterface` constructs payload.

If `ChatInterface` sends `model: "x-ai/grok..."` but backend `llm.py` ignores it and uses `settings.OPENROUTER_MODEL` (which might be deepseek), it should still work but use the wrong model.
If backend uses the model passed in `req.model`, it pass `model="x-ai/grok..."` to `litellm`.
If `litellm` fails (e.g. invalid key for that model, or model 404), it raises exception.
But we saw no logs.

**Hypothesis**: The print flushing fix might rely on `python -u` or refactoring. `sys.stdout.reconfigure(line_buffering=True)` is better.

**Action**:
1. Check `ChatInterface.tsx`.
2. Fix model persistence.
