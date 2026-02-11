import { useState, useEffect, useRef } from 'react';
import { Settings, Plus, History as HistoryIcon } from 'lucide-react';
import { SessionList } from './SessionList';

interface SidebarProps {
    onNewChat?: () => void;
    currentModel: string;
    onOpenSettings: () => void;
    onSelectSession?: (id: string) => void;
    onOpenExplorer?: () => void;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
    terminalEntries?: any[];
    codeEntries?: any[];
}

export const Sidebar: React.FC<SidebarProps> = ({
    onNewChat,
    currentModel,
    onOpenSettings,
    onSelectSession,
    onOpenExplorer,
    usage,
    terminalEntries = [],
    codeEntries = []
}) => {
    const [historyOpen, setHistoryOpen] = useState(false);
    const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
    const wsRef = useRef<WebSocket | null>(null);

    // Refs for auto-scrolling
    const terminalRef = useRef<HTMLDivElement>(null);
    const codeRef = useRef<HTMLDivElement>(null);

    // Auto-scroll terminal to bottom when new entries arrive
    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [terminalEntries]);

    // Auto-scroll code panel
    useEffect(() => {
        if (codeRef.current) {
            codeRef.current.scrollTop = codeRef.current.scrollHeight;
        }
    }, [codeEntries]);

    // WebSocket Log Stream Connection (Ground Truth)
    useEffect(() => {
        const connect = () => {
             // Correctly targeting the backend's logging websocket
             // Use window.location.hostname to be more robust for different network environments
             const backendHost = window.location.hostname === 'localhost' ? 'localhost:8000' : `${window.location.hostname}:8000`;
             const socket = new WebSocket(`ws://${backendHost}/api/v1/ws/logs`);
             wsRef.current = socket;

             socket.onopen = () => {
                 console.log("[Terminal] Connected to backend log stream.");
             };

             socket.onmessage = (event) => {
                 if (event.data) {
                     setTerminalLogs(prev => [...prev, event.data].slice(-1000));
                 }
             };

             socket.onclose = () => {
                 console.log("[Terminal] Disconnected. Retrying in 3s...");
                 setTimeout(connect, 3000);
             };

             socket.onerror = (err) => {
                 console.error("[Terminal] WebSocket error:", err);
             };
        };

        connect();

        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, []);

    // Auto-scroll terminal log
    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [terminalLogs]);

    // Use props directly after routing in App.tsx
    const executionEntries = codeEntries;
    const terminalLogsFromSSE = terminalEntries;

    return (
        <div className="w-[450px] bg-slate-950 h-screen border-r border-slate-800 flex flex-col font-sans text-slate-200 transition-all shadow-2xl z-30">
             {/* Header */}
             <div className="p-3 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center shrink-0">
                <div className="flex items-center gap-3">
                     <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                     <h1 className="text-xs font-bold text-slate-100 tracking-wider">GRAPH RLM <span className="text-slate-500 text-[10px]">v2.0</span></h1>
                </div>
                <div className="flex gap-2">
                     <button
                        onClick={onOpenExplorer}
                        className="text-slate-500 hover:text-emerald-400 p-1 flex items-center gap-1 bg-slate-900 rounded border border-slate-800 px-2"
                        title="Open Full Graph Explorer"
                     >
                        <span className="text-[9px] font-bold uppercase">Session History</span>
                     </button>
                     <button
                        onClick={onNewChat}
                        className="text-slate-500 hover:text-blue-400 p-1 flex items-center gap-1 bg-slate-900 rounded border border-slate-800 px-2"
                        title="Start New Session"
                     >
                        <Plus size={12} />
                        <span className="text-[9px] font-bold uppercase">New Chat</span>
                     </button>
                     <Settings size={14} className="text-slate-500 hover:text-white cursor-pointer ml-1 self-center" onClick={onOpenSettings} />
                </div>
            </div>

            {/* Compact Model Status */}
            <div className="px-3 py-2 border-b border-slate-800 bg-slate-950/50 shrink-0">
                 <div className="flex justify-between items-center text-[10px]">
                      <span className="text-slate-400 font-mono truncate max-w-[200px]">{currentModel}</span>
                      {usage && (
                          <span className="text-slate-600 font-mono">
                              {usage.total_tokens} T
                          </span>
                      )}
                 </div>
            </div>

            {/* MAIN DASHBOARD - Two stacked panels */}
            <div className="flex-1 flex flex-col min-h-0 bg-black/20">

                {/* TOP: Code Execution Results (from replEntries with style='code') */}
                <div className="flex-1 min-h-0 flex flex-col border-b border-slate-700">
                    <div className="px-3 py-1 bg-slate-900/80 text-[10px] font-bold text-blue-400 uppercase tracking-widest flex justify-between">
                        <span>Code Execution</span>
                        <span className="text-slate-600">{executionEntries.length} results</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 font-mono text-[10px] space-y-2 bg-[#0d1117]">
                        {executionEntries.length === 0 && <div className="text-slate-700 italic text-center mt-4">Waiting for code execution...</div>}
                        {executionEntries.map((e, i) => (
                            <div key={i} className="border-l-2 border-blue-900/30 pl-2">
                                <div className="text-blue-300/50 text-[9px] mb-1">
                                    {new Date(e.timestamp).toLocaleTimeString()}
                                    {e.isStreaming && <span className="ml-2 animate-pulse text-blue-500">RUNNING</span>}
                                </div>
                                <pre className="whitespace-pre-wrap text-blue-100/90 break-words">{e.content.replace('[EXECUTION]', '').trim()}</pre>
                            </div>
                        ))}
                    </div>
                </div>

                {/* BOTTOM: Terminal Log (agent events from SSE) */}
                <div className="flex-1 min-h-0 flex flex-col">
                    <div className="px-3 py-1 bg-slate-900/80 text-[10px] font-bold text-emerald-400 uppercase tracking-widest flex justify-between">
                        <span>Terminal Log</span>
                        <span className="text-slate-600">{terminalEntries.length} events</span>
                    </div>
                    <div
                        ref={terminalRef}
                        className="flex-1 overflow-y-auto p-2 font-mono text-[9px] bg-[#0f1216]"
                    >
                        {/* Terminal Log: Now uses BOTH the raw Websocket logs AND the interactive replEntries for total visibility */}
                        {terminalLogs.length === 0 && terminalEntries.length === 0 && <div className="text-slate-700 italic text-center mt-4">Waiting for system logs...</div>}

                        {/* Backend System Logs (uvicorn/bash mirror) from separate WS if still needed */}
                        {terminalLogs.map((log, i) => (
                            <div key={`system-${i}`} className="text-slate-500 leading-tight whitespace-pre-wrap break-words border-b border-slate-950/20 py-0.5">
                                {log}
                            </div>
                        ))}

                        {/* Routed Terminal Events from SSE (Agent Logic) */}
                        {terminalLogsFromSSE.map((entry: any, i: number) => (
                            <div key={`event-${i}`} className={`mb-1 leading-tight flex gap-2 ${entry.type === 'error' ? 'text-red-400' : 'text-slate-500'}`}>
                                <span className="text-slate-700 shrink-0">[{new Date(entry.timestamp).toLocaleTimeString()}]</span>
                                {entry.repl_id && <span className="text-blue-900 shrink-0">({entry.repl_id})</span>}
                                <span className={`shrink-0 uppercase text-[8px] px-1 rounded ${
                                    entry.style === 'report' ? 'bg-blue-900/30 text-blue-400' :
                                    entry.style === 'code' ? 'bg-emerald-900/30 text-emerald-400' :
                                    'bg-slate-900 text-slate-600'
                                }`}>
                                    {entry.style || entry.type}
                                </span>
                                <span className="text-slate-300 break-words flex-1">{entry.content}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Collapsible History */}
            <div className="border-t border-slate-800 bg-slate-900/30 shrink-0">
                 <button
                    onClick={() => setHistoryOpen(!historyOpen)}
                    className="w-full flex items-center justify-between p-3 text-[10px] uppercase font-bold text-slate-500 hover:text-slate-300 transition-colors"
                 >
                    <div className="flex items-center gap-2">
                        <HistoryIcon size={12} />
                        <span>Session History</span>
                    </div>
                    <span className="text-xs">{historyOpen ? '−' : '+'}</span>
                 </button>

                 {historyOpen && (
                     <div className="border-t border-slate-800 max-h-[200px] overflow-hidden bg-slate-950 flex flex-col">
                        <SessionList
                            onSelectSession={(id) => onSelectSession && onSelectSession(id)}
                            className="bg-transparent"
                        />
                     </div>
                 )}
            </div>

             {/* User Profile */}
             <div className="p-3 border-t border-slate-800 bg-slate-950 shrink-0 flex items-center gap-3">
                  <div className="w-6 h-6 rounded bg-gradient-to-br from-blue-900 to-indigo-900 flex items-center justify-center font-bold text-[10px] text-blue-200">TY</div>
                  <div className="text-[10px] text-slate-600">Local Admin</div>
             </div>
        </div>
    );
};
