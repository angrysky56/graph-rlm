
import { useState, useEffect, useRef } from 'react';
import { Layout } from './components/layout/Layout';
import { ReplConsole } from './components/chat/ReplConsole';
import { ChatInput } from './components/chat/ChatInput';

import { v4 as uuidv4 } from 'uuid';
import { api } from './api';

function App() {
  const [sessionId, setSessionId] = useState<string>(() => {
    return localStorage.getItem('NEXUS_SESSION_ID') || uuidv4();
  });

  const [currentModel, setCurrentModel] = useState<string>(''); // Empty initially, load from config
  const [replEntries, setReplEntries] = useState<any[]>([]);

  // Chat Input State (Lifted for Injection)
  const [chatInput, setChatInput] = useState("");

  // Graph State
  const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] }>({ nodes: [], links: [] });
  const [isProcessing, setIsProcessing] = useState(false);


  const abortControllerRef = useRef<AbortController | null>(null);

  // Load Config on Mount
  const refreshConfig = async () => {
    try {
      const cfg = await api.getConfig();
      // Determine active model based on provider
      const provider = cfg.LLM_PROVIDER || 'ollama';
      let active = '';
      if (provider === 'ollama') active = cfg.OLLAMA_MODEL;
      else if (provider === 'openrouter') active = cfg.OPENROUTER_MODEL;
      else if (provider === 'openai') active = cfg.OPENAI_MODEL;

      if (active) setCurrentModel(active);
      return true;
    } catch (e) {
      console.error("Failed to load config (Backend likely starting...)", e);
      return false;
    }
  };

  const loadGraph = async (sid?: string) => {
    try {
      const targetSession = sid || sessionId;
      const data = await api.getGraphState(targetSession);
      if (data && data.nodes) {
        setGraphData(data);
        return true;
      }
    } catch (e) { return false; }
    return false;
  };

  useEffect(() => {
    localStorage.setItem('NEXUS_SESSION_ID', sessionId);
  }, [sessionId]);

  useEffect(() => {
    // Retry loop for initial connection (Backend takes time to spin up MCP)
    let retries = 0;
    let mounted = true;

    const attempt = async () => {
      if (!mounted) return;

      // We check for success boolean now
      const configOk = await refreshConfig();
      await loadGraph();

      // If config loaded (model selected), we consider backend ready.
      // Graph might be empty for new session, so configOk is the main signal.
      if (configOk) {
        return;
      }

      if (retries < 30) {
        retries++;
        // Backoff: 2s, 3s, ... max 10s
        const delay = Math.min(2000 * Math.pow(1.1, retries), 10000);
        setTimeout(attempt, delay);
      }
    };

    attempt();

    return () => { mounted = false; };
  }, []);

  const handleNewChat = () => {
    const newId = uuidv4();
    setSessionId(newId);
    setReplEntries([]);
    setGraphData({ nodes: [], links: [] });
    // Reset usage stats

  };

  const handleSessionSelect = (sid: string) => {
    setSessionId(sid);
    setReplEntries([]); // Clear logs (Future: load logs from DB?)
    loadGraph(sid);
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsProcessing(false);
      setReplEntries(prev => [...prev, { role: 'system', content: '**Stopped by user**', timestamp: Date.now() }]);
    }
  };

  // ... imports

  // Mapping Stream Events to REPL Entries


  // ... (keep handleNewChat as single source of truth at top level)

  const handleExecute = (query: string) => {
    setIsProcessing(true);
    setReplEntries(prev => [...prev, { type: 'input', content: query, timestamp: Date.now() }]);

    const payload = {
      model: currentModel,
      messages: [{ role: 'user', content: query }],
      stream: true,
      session_id: sessionId
    };

    const ctrl = api.streamChat(payload, (event) => {
      if (event.type === 'token') {
        setReplEntries(prev => {
          const last = prev[prev.length - 1];
          if (last && last.type === 'output' && last.isStreaming) {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + event.content }
            ];
          } else {
            return [...prev, { type: 'output', content: event.content, timestamp: Date.now(), isStreaming: true }];
          }
        });

      } else if (event.type === 'thinking') {
        // ...
        // 'Thinking' events (gray text)
        setReplEntries(prev => [...prev, { type: 'info', content: event.content, timestamp: Date.now() }]);

      } else if (event.type === 'code_output_chunk') {
        // Appending streamed code output
        setReplEntries(prev => {
          const last = prev[prev.length - 1];
          // If last entry is streaming code output (we use style='code' to distinguish)
          if (last && last.type === 'output' && last.style === 'code' && last.isStreaming) {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + event.content }
            ];
          } else {
            // Start new code output block
            return [...prev, {
              type: 'output',
              content: event.content, // Start with this chunk
              timestamp: Date.now(),
              style: 'code',
              isStreaming: true
            }];
          }
        });

      } else if (event.type === 'code_output') {
        // Final Code Execution Output (Complete)
        // If we were streaming, we might just mark it done or replace if needed.
        // But backend emits this as the "Final" block with metadata.
        // Let's replace the streaming block with this final one to ensure formatting/correctness
        setReplEntries(prev => {
          const last = prev[prev.length - 1];
          if (last && last.isStreaming && last.style === 'code') {
            // Replace the streaming block with the final complete block
            return [...prev.slice(0, -1), {
              type: 'output',
              content: `[EXECUTION]\n${event.code}\n\n>> ${event.content}`,
              timestamp: Date.now(),
              style: 'code',
              isStreaming: false
            }];
          } else {
            // Just add it if we weren't streaming (fallback)
            return [...prev, {
              type: 'output',
              content: `[EXECUTION]\n${event.code}\n\n>> ${event.content}`,
              timestamp: Date.now(),
              style: 'code'
            }];
          }
        });

      } else if (event.type === 'graph_update') {
        // Keep existing graph logic...
        const { action, node, link } = event.data;
        setGraphData(prev => {
          const newData = { ...prev };
          if (action === 'add_node') {
            if (!newData.nodes.find(n => n.id === node.id)) newData.nodes = [...newData.nodes, node];
          } else if (action === 'add_link') {
            newData.links = [...newData.links, link];
          } else if (action === 'update_node') {
            newData.nodes = newData.nodes.map(n => n.id === node.id ? { ...n, ...node } : n);
          }
          return newData;
        });

      } else if (event.type === 'done') {
        setIsProcessing(false);
        // Mark last streaming entry as done?
        setReplEntries(prev => {
          const last = prev[prev.length - 1];
          if (last && last.isStreaming) {
            return [...prev.slice(0, -1), { ...last, isStreaming: false }];
          }
          return prev;
        });
        abortControllerRef.current = null;
      } else if (event.type === 'error') {
        setReplEntries(prev => [...prev, { type: 'error', content: `Error: ${event.content}`, timestamp: Date.now() }]);
        setIsProcessing(false);
      }
    });

    abortControllerRef.current = ctrl;
  };

  // We need to fetch the full model object to know the provider, because ModelSelector only passes the ID string.
  // Or we modify ModelSelector to pass the object.
  // Modification to Layout required to pass available models map?
  // Let's assume we can infer provider or fetch it.

  // Actually, App.tsx doesn't have the list of models in state to lookup.
  // Simplify: Assuming standard format 'provider/model-name' usually used in IDs?
  // No, OpenRouter IDs are 'vendor/name'. Ollama are 'name'.

  // Strategy: Move model state fetching to App (or context) so we can lookup.
  // For now: Just try to heuristic detecting provider.

  const handleModelSelect = async (modelId: string) => {
    // 1. Optimistic Update
    setCurrentModel(modelId);

    // 2. Persist to Backend
    try {
      const updates: any = {};
      let provider = 'openrouter';

      // Basic Heuristics if we don't have metadata
      if (!modelId.includes('/') && !modelId.includes(':')) {
        // Likely Ollama? Or simple name
        // Check if current provider is ollama?
        // Default to OpenRouter if slashed?
      }

      // BETTER: Keep the current provider unless explicit switch?
      // Actually, if the user used the selector, they picked a specific model which belongs to a specific provider group.
      // But we lost that grouping info here.

      // Let's rely on the ID format for OpenRouter (e.g. google/gemini).
      // Ollama usually has no slash or different format.
      // But OpenRouter includes many providers.

      // Fix: We update settings based on active provider in config if possible,
      // OR we just send the model ID to the fields.
      // The backend `llm.py` uses `OPENROUTER_MODEL` if provider is openrouter.
      // If we don't switch provider, we might set wrong key.

      // Let's just set ALL model keys to this ID to be safe? No.

      // Correct approach: App needs access to the models list to find the provider.
      // But refactoring that is large.
      // Quick fix: Assume OpenRouter for now if it contains '/', else Ollama?

      if (modelId.includes('/')) {
        provider = 'openrouter';
        updates.OPENROUTER_MODEL = modelId;
      } else {
        provider = 'ollama';
        updates.OLLAMA_MODEL = modelId;
      }

      updates.LLM_PROVIDER = provider;

      await api.updateConfig(updates);
      // alert(`Switched to ${modelId}`); // Optional feedback
    } catch (e) {
      console.error("Failed to persist model", e);
    }
  };

  return (
    <Layout
      graphData={graphData}
      onNewChat={handleNewChat}
      currentModel={currentModel}
      onSelectModel={handleModelSelect}
      onRefreshConfig={refreshConfig}
      onInjectContent={(text) => setChatInput(prev => prev + text)}
      onSelectSession={handleSessionSelect}
    >
      <div className="flex h-full relative flex-col">
        {/* REPL Area - Takes full height minus input */}
        <div className="flex-1 overflow-hidden relative">
          <div className="absolute inset-0">
            {/* We render ReplConsole directly. It handles scrolling. */}
            {/* We map replEntries to the props expected by ReplConsole or update ReplConsole to match */}
            <ReplConsole entries={replEntries} />
          </div>
        </div>

        {/* Input Area */}
        <div className="shrink-0">
          <ChatInput
            onSend={handleExecute} // Fixed sig
            onStop={handleStop}
            isProcessing={isProcessing}
            value={chatInput}
            onChange={setChatInput}
          />
        </div>
      </div>
    </Layout>
  );
}

export default App;
