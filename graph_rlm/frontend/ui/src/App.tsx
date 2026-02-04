
import { useState, useEffect, useRef } from 'react';
import { Layout } from './components/layout/Layout';
import { ChatInput } from './components/chat/ChatInput';
import { ChatHistory } from './components/chat/ChatHistory'; // Import ChatHistory
// Scratchpad is now unused here, but kept for Layout prop passing? No, pass string to Layout.
import { GraphExplorer } from './components/graph/GraphExplorer';

import { v4 as uuidv4 } from 'uuid';
import { api } from './api';

function App() {
  const [viewMode, setViewMode] = useState<'explorer' | 'chat'>('chat'); // Default to Chat (Agent View)
  const [sessionId, setSessionId] = useState<string>(() => {
    return localStorage.getItem('NEXUS_SESSION_ID') || uuidv4();
  });

  const [currentModel, setCurrentModel] = useState<string>(''); // Empty initially, load from config
  const [replEntries, setReplEntries] = useState<any[]>([]);
  const [scratchpadText, setScratchpadText] = useState<string>(''); // ACTUAL scratchpad text from agent

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

  const loadGraph = async (sid?: string | null) => {
    try {
      // If sid is explicitly null, we want global graph (no filter).
      // If sid is string, use it.
      // If sid is undefined, fallback to current sessionId (refresh).
      const targetSession = sid === undefined ? sessionId : sid;

      // Ensure we query for *something* if we are in a session
      // If targetSession is empty string/null, we get global graph.
      const queryId = targetSession || undefined;

      console.log(`[App] Loading Graph for Session: ${queryId || 'GLOBAL'}`);
      const data = await api.getGraphState(queryId);
      if (data && data.nodes) {
        setGraphData(data);
        return true;
      }
      return true;
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
      // Fetch Graph initially
      // Always load the graph for the current session to ensure consistency.
      // If the user wants a "Global View", we can add a specific toggle later.
      const graphOk = await loadGraph(sessionId);

      // Initial Status Fetch (No Polling)
      try {
          if (sessionId) {
            await api.getSystemStatus(sessionId);
          }
      } catch (e) { console.error("Initial status fetch failed", e); }


      // Ensure BOTH are ready before stopping retries
      if (configOk && graphOk) {
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

    // No Polling - Updates are pushed via SSE

    return () => {
        mounted = false;
    };
  }, [sessionId]);

  const handleNewChat = () => {
    const newId = uuidv4();
    setSessionId(newId);
    setReplEntries([]);
    setScratchpadText(''); // Clear scratchpad
    setGraphData({ nodes: [], links: [] });
    // Reset usage stats

  };

  const handleSessionSelect = async (sid: string) => {
    setSessionId(sid);
    setReplEntries([]); // Clear current while loading
    setScratchpadText(''); // Clear while loading
    loadGraph(sid); // Restore explicit load to handle view mode switches

    // Note: Scratchpad text will be populated when agent runs again
    // The scratchpad_text event provides the actual context

    try {
      const history = await api.getHistory(sid);
      if (history && Array.isArray(history)) {
        const entries = history.map((msg: any) => {
          // Robust content mapping using new backend fields
          let finalContent = msg.content || "";
          let finalStyle: 'code' | 'thinking' | 'trace' | 'success' | 'error' | undefined = undefined;
          let finalType: 'input' | 'output' | 'info' | 'error' = msg.role === 'user' ? 'input' : 'output';

          // 1. Detect Final Answer (Simple heuristic if backend marks it)
          if (msg.status === 'success' && !msg.repl_id && msg.result) {
               // Could be final answer
          }

          // 2. Detect Code Execution
          // If we have a repl_id, OR prompt starts with code indicators
          // msg.content often has "Thought: ... Code: ..." or similar if raw
          const isCode = msg.repl_id || (typeof msg.content === 'string' && (msg.content.includes("def ") || msg.content.includes("import "))) || msg.execution_summary;

          if (isCode && msg.role !== 'user') {
             finalStyle = 'code';
             // If we have a result, append it neatly
             if (msg.execution_summary) {
                 finalContent += `\n\n> **Result:**\n${msg.execution_summary}`;
             } else if (msg.result) {
                 finalContent += `\n\n> **Result:**\n${msg.result}`;
             }
             if (msg.repl_id) {
                 finalContent = `[REPL: ${msg.repl_id}]\n${finalContent}`;
             }
          }
          // 3. Simple Result Append (if not code but has result)
          else if (msg.result && msg.role !== 'user') {
             finalContent += `\n\n> **Result:**\n${msg.result}`;
          }

          // 4. Error Status
          if (msg.status === 'error') {
              finalType = 'error';
              finalStyle = 'error';
          }

          return {
            type: finalType,
            content: finalContent,
            timestamp: msg.created_at ? new Date(msg.created_at).getTime() : Date.now(),
            style: finalStyle,
            repl_id: msg.repl_id
          };
        });
        setReplEntries(entries);
      }
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  };

  const handleStop = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    await api.stopGeneration(); // Tell backend to stop
    setIsProcessing(false);
    setReplEntries(prev => [...prev, { role: 'system', content: '**Stopped by user**', timestamp: Date.now() }]);
  };

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
          // Only append to last entry if it's a generic output (not code) and still streaming
          if (last && last.type === 'output' && last.isStreaming && last.style !== 'code') {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + event.content }
            ];
          } else {
            return [...prev, { type: 'output', content: event.content, timestamp: Date.now(), isStreaming: true }];
          }
        });

      } else if (event.type === 'thinking') {
        // ALL thinking events go to Live REPL (bottom left) UNFILTERED
        // This is the raw terminal output - no filtering
        setReplEntries(prev => [...prev, {
          type: 'info',
          content: event.content || '',
          timestamp: Date.now(),
          style: 'thinking'
        }]);

      } else if (event.type === 'code_output_chunk') {
        // Appending streamed code output
        setReplEntries(prev => {
          const last = prev[prev.length - 1];
          // If last entry is streaming code output
          if (last && last.type === 'output' && last.style === 'code' && last.isStreaming) {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + event.content }
            ];
          } else {
            // Start new code output block
            return [...prev, {
              type: 'output',
              content: event.content,
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
              content: `[EXECUTION] (REPL: ${event.data?.repl_id || 'unknown'})\n${event.code}\n\n>> ${event.content}`,
              timestamp: Date.now(),
              style: 'code',
              isStreaming: false
            }];
          } else {
            // Just add it if we weren't streaming (fallback)
            return [...prev, {
              type: 'output',
              content: `[EXECUTION] (REPL: ${event.data?.repl_id || 'unknown'})\n${event.code}\n\n>> ${event.content}`,
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
            console.log("[GraphUpdate] Adding Node:", node);
            if (!newData.nodes.find(n => n.id === node.id)) newData.nodes = [...newData.nodes, node];
          } else if (action === 'add_link') {
            console.log("[GraphUpdate] Adding Link:", link);
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
      } else if (event.type === 'trace') {
        // Handle Trace Logs (System Observability)
        setReplEntries(prev => [...prev, {
            type: 'info',
            content: event.content,
            timestamp: Date.now(),
            style: 'trace' // New style for trace logs
        }]);

      } else if (event.type === 'warning') {
        setReplEntries(prev => [...prev, { type: 'error', content: `Warning: ${event.content}`, timestamp: Date.now() }]);
      } else if (event.type === 'error') {
        setReplEntries(prev => [...prev, { type: 'error', content: `Error: ${event.content}`, timestamp: Date.now() }]);
        setIsProcessing(false);
      } else if (event.type === 'scratchpad_text') {
        // Store the ACTUAL scratchpad text the agent sees
        // This is the verbatim output of build_scratchpad()
        setScratchpadText(event.content || '');
      } else if (event.type === 'active_thought') {
        // Note: We no longer track individual thoughts in state
        // The scratchpad_text event gives us the complete context
        console.debug('[AGENT] Active thought:', event.data?.id);

      } else if (event.type === 'scratchpad_update') {
        // Scratchpad will be updated by the next scratchpad_text event
        console.debug('[AGENT] Scratchpad update pending');

      } else if (event.type === 'answer' || event.type === 'final_answer') {
        // Final Answer Event - show in sidebar
        const answerContent = event.content;

        // Show final answer in sidebar
        setReplEntries(prev => [...prev, {
          type: 'output',
          content: `✅ **Final Answer:**\n${answerContent}`,
          timestamp: Date.now(),
          style: 'success'
        }]);

        setIsProcessing(false);
      }
    });

    abortControllerRef.current = ctrl;
  };



  // Render Explorer Mode
  if (viewMode === 'explorer') {
      return (
          <GraphExplorer
              graphData={graphData}
              activeSessionId={sessionId}
              onSelectSession={handleSessionSelect}
              onSwitchToChat={() => {
                  setViewMode('chat');
                  loadGraph(sessionId); // Restore Session Context
              }}
              onNodeClick={(node) => {
                  console.log("Node clicked", node);
              }}
              onShowFullGraph={() => {
                  loadGraph(null); // Load Global Graph
              }}
          />
      );
  }

  return (
    <Layout
      graphData={graphData}
      onNewChat={handleNewChat}
      currentModel={currentModel}
      onRefreshConfig={refreshConfig}
      onInjectContent={(text) => setChatInput(prev => prev + text)}
      onSelectSession={handleSessionSelect}
      onOpenExplorer={() => {
        setViewMode('explorer');
        loadGraph(null); // Force Global Load
      }}
      replEntries={replEntries}
      scratchpadText={scratchpadText} // Pass to Layout -> RightSidebar
    >

      <div className="flex h-full relative flex-col">
        {/* Center: Chat History (Was Scratchpad) */}
        <div className="flex-1 min-h-0 relative flex flex-col">
          <ChatHistory entries={replEntries} />
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
