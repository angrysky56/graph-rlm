import { Settings, Plus, Activity, BookOpen, Hash, Code, Copy, Check } from 'lucide-react';
import { useState } from 'react';

interface SidebarProps {
    onNewChat?: () => void;
    currentModel: string;
    onOpenSettings: () => void;
    onOpenExplorer?: () => void;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
    selectedNode: any | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
    onNewChat,
    currentModel,
    onOpenSettings,
    onOpenExplorer,
    usage,
    selectedNode
}) => {
    const [copiedSection, setCopiedSection] = useState<string | null>(null);

    const handleCopy = (text: string, section: string) => {
        navigator.clipboard.writeText(text);
        setCopiedSection(section);
        setTimeout(() => setCopiedSection(null), 2000);
    };

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
                 </div>
            </div>

            {/* MAIN DASHBOARD - Node Explorer */}
            <div className="flex-1 flex flex-col min-h-0 bg-[#0a0a0a]">
                <div className="px-3 py-1.5 bg-slate-900/80 text-[10px] font-bold text-indigo-400 uppercase tracking-widest border-b border-slate-800 flex items-center gap-2">
                    <Activity size={12} />
                    <span>Node Explorer</span>
                </div>

                {selectedNode ? (
                    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
                        <div className="bg-slate-900/30 border border-slate-800 rounded-lg p-3">
                            <div className="flex items-center gap-2 mb-3 text-slate-400">
                                <Hash size={14} className="text-indigo-400" />
                                <span className="text-xs font-mono font-bold text-slate-200">{selectedNode.id}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 text-[10px] uppercase font-bold tracking-wider">
                                <span className={`px-2 py-1 rounded ${
                                    selectedNode.status === 'success' ? 'bg-emerald-900/40 text-emerald-400 border border-emerald-800/50' :
                                    selectedNode.status === 'error' || selectedNode.status === 'failed' ? 'bg-red-900/40 text-red-400 border border-red-800/50' :
                                    selectedNode.status === 'running' ? 'bg-blue-900/40 text-blue-400 border border-blue-800/50' :
                                    'bg-slate-800 border border-slate-700 text-slate-400'
                                }`}>
                                    {selectedNode.status || 'UNKNOWN'}
                                </span>
                                {(selectedNode.turn_id !== undefined || selectedNode.round_id !== undefined) && (
                                    <span className="bg-slate-800/50 border border-slate-700 text-slate-400 px-2 py-1 rounded">
                                        Round {selectedNode.round_id || '?'} / Turn {selectedNode.turn_id ?? '?'}
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1.5 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                                    <BookOpen size={12} /> Prompt Context
                                </div>
                                {selectedNode.prompt && (
                                    <button
                                        onClick={() => handleCopy(selectedNode.prompt, 'prompt')}
                                        className="text-slate-500 hover:text-indigo-400 transition-colors"
                                        title="Copy Prompt"
                                    >
                                        {copiedSection === 'prompt' ? <Check size={12} /> : <Copy size={12} />}
                                    </button>
                                )}
                            </div>
                            <div className="bg-[#0d1117] border border-slate-800 rounded p-3 text-[11px] leading-relaxed text-slate-300 font-mono whitespace-pre-wrap break-words max-h-64 overflow-y-auto custom-scrollbar shadow-inner">
                                {selectedNode.prompt || 'No prompt recorded.'}
                            </div>
                        </div>

                        <div className="space-y-2">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1.5 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                                    <Code size={12} /> Execution Result
                                </div>
                                {selectedNode.result && (
                                    <button
                                        onClick={() => handleCopy(selectedNode.result, 'result')}
                                        className="text-slate-500 hover:text-indigo-400 transition-colors"
                                        title="Copy Result"
                                    >
                                        {copiedSection === 'result' ? <Check size={12} /> : <Copy size={12} />}
                                    </button>
                                )}
                            </div>
                            <div className="bg-[#0d1117] border border-slate-800 rounded p-3 text-[11px] leading-relaxed text-slate-300 font-mono whitespace-pre-wrap break-words max-h-64 overflow-y-auto custom-scrollbar shadow-inner">
                                {selectedNode.result || 'No numeric or string result recorded.'}
                            </div>
                        </div>

                        {(selectedNode.sheaf_score !== undefined || selectedNode.omcd_score !== undefined || selectedNode.repe_shakiness !== undefined) && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-1.5 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                                    <Activity size={12} /> Observability Metrics
                                </div>
                                <div className="bg-slate-900/30 border border-slate-800 rounded p-3 text-[10px] font-mono grid grid-cols-2 gap-3">
                                    {selectedNode.sheaf_score !== undefined && selectedNode.sheaf_score !== null && (
                                        <div className="flex flex-col gap-1 p-2 bg-slate-900/50 rounded border border-slate-800/50">
                                            <span className="text-slate-500 uppercase">Sheaf Consistency</span>
                                            <span className="text-emerald-400 text-sm font-bold">{Number(selectedNode.sheaf_score).toFixed(3)}</span>
                                        </div>
                                    )}
                                    {selectedNode.omcd_score !== undefined && selectedNode.omcd_score !== null && (
                                        <div className="flex flex-col gap-1 p-2 bg-slate-900/50 rounded border border-slate-800/50">
                                            <span className="text-slate-500 uppercase">oMCD Stop Prob</span>
                                            <span className="text-blue-400 text-sm font-bold">{Number(selectedNode.omcd_score).toFixed(3)}</span>
                                        </div>
                                    )}
                                    {selectedNode.repe_shakiness !== undefined && selectedNode.repe_shakiness !== null && (
                                        <div className="flex flex-col gap-1 p-2 bg-slate-900/50 rounded border border-slate-800/50">
                                            <span className="text-slate-500 uppercase">Shakiness</span>
                                            <span className="text-yellow-400 text-sm font-bold">{Number(selectedNode.repe_shakiness).toFixed(3)}</span>
                                        </div>
                                    )}
                                    {selectedNode.repe_evasion !== undefined && selectedNode.repe_evasion !== null && (
                                        <div className="flex flex-col gap-1 p-2 bg-slate-900/50 rounded border border-slate-800/50">
                                            <span className="text-slate-500 uppercase">Evasion</span>
                                            <span className="text-red-400 text-sm font-bold">{Number(selectedNode.repe_evasion).toFixed(3)}</span>
                                        </div>
                                    )}
                                    {selectedNode.h0_rank !== undefined && selectedNode.h0_rank !== null && (
                                        <div className="flex flex-col gap-1 p-2 bg-slate-900/50 rounded border border-slate-800/50">
                                            <span className="text-slate-500 uppercase">H0 Rank</span>
                                            <span className="text-purple-400 text-sm font-bold">{selectedNode.h0_rank}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center p-8 text-slate-500 h-full text-center">
                        <Activity size={32} className="mb-4 opacity-30 text-indigo-400" />
                        <p className="text-[11px] uppercase tracking-wider font-bold mb-3 text-slate-400">Awaiting Sub-Selection</p>
                        <p className="text-[10px] leading-relaxed max-w-[250px]">To inspect historical state, click any structural node embedded in the active Topological Graph (Right Panel).</p>
                    </div>
                )}
            </div>

            {/* Token Usage Display */}
            <div className="border-t border-slate-800 bg-slate-900/30 shrink-0 p-3">
                 <div className="w-full flex items-center justify-between text-[10px] uppercase font-bold text-slate-500">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-500/50 animate-pulse"></span>
                        <span>Token Usage</span>
                    </div>
                    {usage ? (
                        <div className="text-right" title={`Prompt: ${usage.prompt_tokens} | Completion: ${usage.completion_tokens}`}>
                            <span className="text-emerald-400 font-mono text-xs block">{usage.total_tokens.toLocaleString()}</span>
                            <span className="text-[8px] text-slate-600 lowercase">total</span>
                        </div>
                    ) : (
                        <span className="text-slate-700 italic">--</span>
                    )}
                 </div>
                 {usage && (
                    <div className="flex gap-1 mt-1 h-1 w-full bg-slate-800 rounded-full overflow-hidden opacity-50">
                        <div
                            className="bg-blue-500/50 h-full transition-all duration-500"
                            style={{ width: `${(usage.prompt_tokens / usage.total_tokens) * 100}%` }}
                        />
                        <div
                            className="bg-emerald-500/50 h-full transition-all duration-500"
                            style={{ width: `${(usage.completion_tokens / usage.total_tokens) * 100}%` }}
                        />
                    </div>
                 )}
            </div>

             {/* User Profile */}
             <div className="p-3 border-t border-slate-800 bg-slate-950 shrink-0 flex items-center gap-3">
                  <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-900 to-purple-900 flex items-center justify-center font-bold text-[10px] text-indigo-200">TY</div>
                  <div className="text-[10px] text-slate-600">Local Admin</div>
             </div>
        </div>
    );
};
