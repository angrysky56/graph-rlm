
import { useState, useEffect, useRef, useCallback } from 'react';
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
  const [chatEntries, setChatEntries] = useState<any[]>([]);
  const [scratchpadText, setScratchpadText] = useState<string>(''); // ACTUAL scratchpad text from agent
  const [selectedNode, setSelectedNode] = useState<any>(null);

  // Chat Input State (Lifted for Injection)
  const [chatInput, setChatInput] = useState("");

  // Graph State
  const [graphData, setGraphData] = useState<{ nodes: any[], links: any[] }>({ nodes: [], links: [] });
  const [isProcessing, setIsProcessing] = useState(false);
  const [tokenUsage, setTokenUsage] = useState<any>(undefined);


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

  const handleNewChat = useCallback(() => {
    const newId = uuidv4();
    setSessionId(newId);
    setChatEntries([]);
    setSelectedNode(null);
    setScratchpadText(''); // Clear scratchpad
    setGraphData({ nodes: [], links: [] });
  }, []);

  const handleSessionSelect = useCallback(async (sid: string) => {
    setSessionId(sid);
    setChatEntries([]);
    setSelectedNode(null);
    setScratchpadText(''); // Clear while loading
    loadGraph(sid); // Restore explicit load to handle view mode switches

    try {
      const history = await api.getHistory(sid);
      if (history && Array.isArray(history)) {
        const entries = history.map((msg: any) => {
          let finalContent = msg.content || "";
          let finalStyle: 'code' | 'thinking' | 'trace' | 'success' | 'error' | undefined = undefined;
          let finalType: 'input' | 'output' | 'info' | 'error' = msg.role === 'user' ? 'input' : 'output';

          if (msg.status === 'success' && !msg.repl_id && msg.result) {
               finalStyle = 'success';
          }

          const isCodeResult = !!(msg.repl_id || msg.execution_summary);

          if (isCodeResult && msg.role !== 'user') {
             finalStyle = 'code';
             if (msg.execution_summary) {
                 finalContent += `\n\n> **Result:**\n${msg.execution_summary}`;
             } else if (msg.result) {
                 finalContent += `\n\n> **Result:**\n${msg.result}`;
             }
             if (msg.repl_id) {
                 finalContent = `[REPL: ${msg.repl_id}]\n${finalContent}`;
             }
          }
          else if (msg.result && msg.role !== 'user') {
             if (msg.status === 'success') finalStyle = 'success';
             finalContent += `\n\n> **Summary:**\n${msg.result}`;
          }

          if (msg.status === 'error') {
              finalType = 'error';
              finalStyle = 'error';
          }

          return {
            type: finalType,
            content: finalContent,
            timestamp: msg.created_at ? new Date(msg.created_at).getTime() : Date.now(),
            style: finalStyle as any,
            repl_id: msg.repl_id
          };
        });

        // Push ALL relevant entries into the chat thread directly.
        // The ChatHistory component will filter and accordion them appropriately.
        setChatEntries(entries.filter((e: any) => e.style !== 'trace'));
      }
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, [loadGraph]);

  const handleStop = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    await api.stopGeneration();
    setIsProcessing(false);
    setChatEntries(prev => [...prev, { role: 'system', content: '**Stopped by user**', timestamp: Date.now() }]);
  }, []);

  const handleExecute = useCallback((query: string) => {
    setIsProcessing(true);
    setChatEntries(prev => [...prev, { type: 'input', content: query, timestamp: Date.now() }]);

    const payload = {
      model: currentModel,
      messages: [{ role: 'user', content: query }],
      stream: true,
      session_id: sessionId
    };

    const ctrl = api.streamChat(payload, (event) => {
      let isSystemic = false;

      if (event.type === 'graph_update') {
        isSystemic = true;
        const { action, node, link } = event.data;
        if (action === 'add_node' || action === 'add_link' || action === 'update_node') {
          setGraphData(prev => {
            const newData = { ...prev };
            if (action === 'add_node') {
              if (!newData.nodes.find((n: any) => n.id === node.id)) newData.nodes = [...newData.nodes, node];
            } else if (action === 'add_link') {
              newData.links = [...newData.links, link];
            } else if (action === 'update_node') {
              newData.nodes = newData.nodes.map((n: any) => n.id === node.id ? { ...n, ...node } : n);
            }
            return newData;
          });
        }
      } else if (event.type === 'scratchpad_text') {
        isSystemic = true;
        setScratchpadText(String(event.content || ''));
      } else if (event.type === 'done') {
        isSystemic = true;
        setIsProcessing(false);
        setChatEntries(prev => {
          const last = prev[prev.length - 1];
          return last && last.isStreaming ? [...prev.slice(0, -1), { ...last, isStreaming: false }] : prev;
        });
      } else if (event.type === 'token_usage') {
        isSystemic = true;
        setTokenUsage(event.data);
      }

      if (isSystemic && event.type !== 'done') return;

      const safeContent = String(event.content || '');

      if (event.ui_target === 'CHAT_RESPONSE') {
        if (event.type === 'token') {
          setChatEntries(prev => {
            const last = prev[prev.length - 1];
            if (last && last.isStreaming) {
              return [...prev.slice(0, -1), { ...last, content: last.content + safeContent }];
            } else {
              return [...prev, { type: 'output', content: safeContent, timestamp: Date.now(), isStreaming: true }];
            }
          });
        } else if (safeContent || event.type === 'RLM_FINAL_RESPONSE') {
          let messageStyle: string | undefined = undefined;
          if (event.type === 'RLM_FINAL_RESPONSE') {
            messageStyle = 'success';
          } else if (event.type === 'thinking') {
            messageStyle = 'thinking';
          } else if (event.type === 'warning' || event.type === 'error') {
            messageStyle = 'error';
          }
          setChatEntries(prev => [...prev, {
            type: 'output',
            content: safeContent,
            timestamp: Date.now(),
            style: messageStyle
          }]);
        }
      }

      else if (event.type === 'repe' || event.type === 'sheaf' || event.type === 'monitor') {
        setChatEntries(prev => [...prev, {
            type: 'output',
            role: 'system',
            content: `📊 Monitor [${event.type.toUpperCase()}]: ${event.message || safeContent}`,
            timestamp: Date.now(),
            style: 'monitor',
            metrics: event.metrics || event.data
        }]);
      }

      else if (event.ui_target === 'CODE_RESULT') {
        // Combine code + execution result in a single code block
        const codeBlock = event.code ? `\`\`\`python\n${event.code}\n\`\`\`\n` : '';
        const resultBlock = event.content ? `**Execution Result:**\n\`\`\`\n${event.content}\n\`\`\`` : '';
        const combined = codeBlock + resultBlock;
        setChatEntries(prev => [...prev, {
            type: 'output',
            content: combined || '(no output)',
            timestamp: Date.now(),
            style: 'code',
            repl_id: event.repl_id,
            isStreaming: event.type === 'code_output_chunk'
        }]);
      }
    });

    abortControllerRef.current = ctrl;
  }, [currentModel, sessionId]);



  const onInjectContent = useCallback((text: string) => {
    console.log("App: onInjectContent called", text);
    setChatInput(prev => prev + text);
  }, []);

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
      onInjectContent={onInjectContent}
      onSelectSession={handleSessionSelect}
      onOpenExplorer={() => {
        setViewMode('explorer');
        loadGraph(null); // Force Global Load
      }}
      scratchpadText={scratchpadText} // Pass to Layout -> RightSidebar
      usage={tokenUsage}
      selectedNode={selectedNode}
      onNodeSelect={setSelectedNode}
    >

      <div className="flex h-full relative flex-col">
        {/* Center: Chat History (Was Scratchpad) */}
        <div className="flex-1 min-h-0 relative flex flex-col">
          <ChatHistory entries={chatEntries} />
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
