import React, { useState, useEffect } from 'react';
import { ModelStatus } from './ModelStatus';
import { Settings, Plus, History as HistoryIcon } from 'lucide-react';
import { type Model, api } from '../../api';

interface SidebarProps {
    onNewChat?: () => void;
    currentModel: string;
    onSelectModel: (model: Model) => void;
    onOpenSettings: () => void;
    onSelectSession?: (id: string) => void;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

export const Sidebar: React.FC<SidebarProps> = ({
    onNewChat,
    currentModel,
    onOpenSettings,
    onSelectSession,
    usage
}) => {
    const [models, setModels] = useState<Model[]>([]);
    const [sessions, setSessions] = useState<any[]>([]);

    useEffect(() => {
        let isStopped = false;

        const fetchModels = async () => {
            let retries = 0;
            const max = 30; // Increased retries
            while (retries < max && !isStopped) {
                try {
                    const data = await api.listModels();
                    // Keep retrying if empty list (assuming backend has at least 1 model)
                    if (data && data.length > 0) {
                        setModels(data);
                        break;
                    }
                } catch (e) { }
                retries++;
                // Exponential backoff
                await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(1.2, retries), 5000)));
            }
        };

        const fetchSessions = async () => {
            let retries = 0;
            const max = 30;
            while (retries < max && !isStopped) {
                try {
                    const data = await api.getSessions();
                    // Sessions CAN be empty, but if we error we retry.
                    // If we get specific error or empty on first try, maybe wait?
                    // Actually, getting sessions is less critical than models.
                    // But let's assume if it works, it works.
                    if (Array.isArray(data)) {
                        setSessions(data);
                        // If we got a valid array, even empty, we stop.
                        break;
                    }
                } catch (e) { }
                retries++;
                await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(1.2, retries), 5000)));
            }
        };

        fetchModels();
        fetchSessions();

        return () => { isStopped = true; };
    }, []);

    return (
        <div className="w-[400px] bg-slate-950 h-screen border-r border-slate-800 flex flex-col font-sans text-slate-200 transition-all">
            {/* Header */}
            <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center shrink-0">
                <h1 className="text-sm font-bold text-slate-100 tracking-wider flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    GRAPH RLM
                </h1>
                <Settings
                    size={14}
                    className="text-slate-500 hover:text-white cursor-pointer transition-colors"
                    onClick={onOpenSettings}
                />
            </div>

            {/* Model Status Section */}
            <div className="p-4 border-b border-slate-800 space-y-3 shrink-0">
                <div className="flex justify-between items-center text-[10px] text-slate-500 uppercase tracking-widest font-bold">
                    <div className="flex items-center gap-2">
                        <span>Reasoning Engine</span>
                        {onNewChat && (
                            <button onClick={onNewChat} className="hover:text-blue-400 transition-colors" title="New Session">
                                <Plus size={10} />
                            </button>
                        )}
                    </div>
                </div>

                <ModelStatus
                    model={models.find(m => m.id === currentModel) || { id: currentModel, name: currentModel, context_length: 0, supports_tools: false, pricing: { prompt: '', completion: '' } }}
                    usage={usage}
                />
            </div>

            {/* History Section (Live Sessions) */}
            <div className="flex-1 flex flex-col min-h-0">
                <div className="p-4 pb-2 text-[10px] text-slate-500 uppercase tracking-widest font-bold flex items-center gap-2">
                    <HistoryIcon size={10} />
                    Recent Sessions
                </div>

                <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-1 scrollbar-thin scrollbar-thumb-slate-800">
                    {sessions.length === 0 && (
                        <div className="p-4 text-center text-xs text-slate-600 italic">No history found.</div>
                    )}
                    {sessions.map(s => (
                        <div key={s.id} onClick={() => onSelectSession && onSelectSession(s.id)} className="p-3 bg-slate-900/40 border-transparent border hover:border-slate-800 rounded cursor-pointer transition-colors group">
                            <div className="text-xs text-slate-300 font-medium truncate">{s.title || "Untitled Session"}</div>
                            <div className="text-[10px] text-slate-600 mt-1 flex justify-between">
                                <span>{new Date(s.created_at || Date.now()).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* User Profile / Status Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-900/30 shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center font-bold text-xs">
                        TY
                    </div>
                    <div>
                        <div className="text-xs font-bold text-slate-200">System Admin</div>
                        <div className="text-[10px] text-slate-500">Connected to Localhost</div>
                    </div>
                </div>
            </div>
        </div>
    );
};
