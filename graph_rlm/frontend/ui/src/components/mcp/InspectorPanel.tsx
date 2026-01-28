import React, { useState, useEffect } from 'react';
import { api } from '../../api';
import { Server, Terminal, Box, Command, RefreshCw, AlertCircle, ChevronRight } from 'lucide-react';

interface InspectorPanelProps {
    onInjectContent: (text: string) => void;
}

interface McpServer {
    name: string;
    enabled: boolean;
    configured: boolean;
    tools: string[];
    error?: string;
}

interface Skill {
    name: string;
    description?: string;
    version?: number;
    tags?: string[];
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({ onInjectContent }) => {
    const [activeTab, setActiveTab] = useState<'tools' | 'skills'>('tools');
    const [servers, setServers] = useState<McpServer[]>([]);
    const [skills, setSkills] = useState<Skill[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set());

    const toggleServer = (name: string) => {
        const next = new Set(expandedServers);
        if (next.has(name)) next.delete(name);
        else next.add(name);
        setExpandedServers(next);
    };

    const fetchData = async () => {
        setLoading(true);
        setError(null);

        const fetchOnce = async () => {
            try {
                // Fetch Servers
                const mcpStatus = await api.getMcpStatus();
                if (mcpStatus.error) {
                    throw new Error(mcpStatus.error || "Failed to fetch MCP status");
                }
                setServers(mcpStatus.servers || []);
                const skillsList = await api.getSkills();
                setSkills(skillsList || []);

                setLoading(false);

                // Auto-Retry if we see discovery persistence errors (e.g. startup in progress)
                // This ensures we auto-refresh once the tools are generated.
                const hasPendingDiscovery = mcpStatus.servers?.some((s: McpServer) =>
                    s.error && s.error.includes("Tool wrapper not found")
                );

                // Also retry if empty (maybe startup?)
                const isEmpty = !mcpStatus.servers || mcpStatus.servers.length === 0;

                if (hasPendingDiscovery || isEmpty) {
                     return false; // Continue polling
                }

                if (hasPendingDiscovery) {
                     return false; // Continue polling
                }

                return true;
            } catch (e) {
                console.error(e);
                return false;
            }
        };

        const runRetryLoop = async () => {
            let retries = 0;
            const max = 30; // 30 retries
            while (retries < max) {
                const success = await fetchOnce();
                if (success) break;
                retries++;
                // Exponential-ish backoff
                await new Promise(r => setTimeout(r, Math.min(1000 * Math.pow(1.5, retries), 10000)));
            }
            if (loading) setLoading(false);
        };

        runRetryLoop();
    };

    useEffect(() => {
        fetchData();
    }, []);

    // Helper to sanitize python name
    const sanitize = (name: string) => name.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase();

    return (
        <div className="flex flex-col h-full bg-slate-950 text-slate-200 font-sans text-xs">
            {/* Tabs */}
            <div className="flex border-b border-slate-800">
                <button
                    onClick={() => setActiveTab('tools')}
                    className={`flex-1 py-2 text-center font-bold tracking-wider transition-colors flex items-center justify-center gap-2 ${activeTab === 'tools' ? 'text-blue-400 bg-slate-900/50 border-b-2 border-blue-500' : 'text-slate-500 hover:text-slate-300'
                        }`}
                >
                    <Terminal size={12} /> TOOLS
                </button>
                <button
                    onClick={() => setActiveTab('skills')}
                    className={`flex-1 py-2 text-center font-bold tracking-wider transition-colors flex items-center justify-center gap-2 ${activeTab === 'skills' ? 'text-purple-400 bg-slate-900/50 border-b-2 border-purple-500' : 'text-slate-500 hover:text-slate-300'
                        }`}
                >
                    <Box size={12} /> SKILLS
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-slate-800">
                {loading && (
                    <div className="flex items-center justify-center p-4 text-slate-500 gap-2">
                        <RefreshCw size={14} className="animate-spin" /> Loading...
                    </div>
                )}

                {error && (
                    <div className="p-3 bg-red-900/20 border border-red-900/50 rounded mb-2 text-red-400 flex flex-col gap-1">
                        <div className="flex items-center gap-2 font-bold"><AlertCircle size={12} /> Error</div>
                        <div className="opacity-80 break-words">{error}</div>
                        <button onClick={fetchData} className="text-xs underline mt-1 hover:text-white">Retry</button>
                    </div>
                )}

                {!loading && !error && activeTab === 'tools' && (
                    <div className="space-y-1">
                        {servers.length === 0 && <div className="text-center text-slate-600 italic py-4">No Detected Servers</div>}

                        {servers.map(server => {
                            const isExpanded = expandedServers.has(server.name);
                            return (
                                <div key={server.name} className="border-b border-slate-900 last:border-0">
                                    <button
                                        onClick={() => toggleServer(server.name)}
                                        className="w-full flex items-center gap-2 px-2 py-2 text-slate-400 font-bold uppercase tracking-wider text-[10px] hover:bg-slate-900/50 transition-colors"
                                    >
                                        <div className={`transition-transform duration-200 ${isExpanded ? 'rotate-90' : ''}`}>
                                            <ChevronRight size={10} />
                                        </div>
                                        <Server size={10} className={server.enabled ? "text-blue-500" : "text-slate-600"} />
                                        <span>{server.name}</span>

                                        <div className="ml-auto flex items-center gap-2">
                                            {server.tools.length > 0 && (
                                                <span className="text-[9px] bg-slate-900 text-slate-600 px-1.5 py-0.5 rounded-full">
                                                    {server.tools.length}
                                                </span>
                                            )}
                                            {server.error && <span className="text-red-500" title={server.error}><AlertCircle size={10} /></span>}
                                        </div>
                                    </button>

                                    {isExpanded && (
                                        <div className="bg-slate-900/20 py-1 px-2 space-y-0.5 border-l-2 border-slate-800 ml-3 mb-2">
                                            {server.tools.length === 0 && !server.error && (
                                                <div className="text-slate-700 italic text-[10px] py-1">No tools available</div>
                                            )}
                                            {server.tools.map(tool => (
                                                <button
                                                    key={tool}
                                                    onClick={() => onInjectContent(`import mcp_tools.${sanitize(server.name)} as ${sanitize(server.name)}\n${sanitize(server.name)}.${sanitize(tool)}() `)}
                                                    className="w-full text-left px-2 py-1.5 rounded hover:bg-blue-500/10 hover:text-blue-300 text-slate-400 transition-colors flex items-center justify-between group"
                                                    title="Click to insert"
                                                >
                                                    <span className="font-mono truncate">{tool}</span>
                                                    <Command size={9} className="opacity-0 group-hover:opacity-50" />
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                {!loading && !error && activeTab === 'skills' && (
                    <div className="space-y-2">
                        {skills.length === 0 && <div className="text-center text-slate-600 italic py-4">No Skills Found</div>}

                        {skills.map(skill => (
                            <button
                                key={skill.name}
                                onClick={() => onInjectContent(`run_skill("${skill.name}")`)}
                                className="w-full text-left p-2 bg-slate-900/30 hover:bg-slate-800 rounded border border-transparent hover:border-purple-900/30 transition-all group"
                            >
                                <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-slate-300 group-hover:text-purple-300 font-mono">{skill.name}</span>
                                    <span className="text-[9px] bg-slate-900 text-slate-500 px-1 rounded">v{skill.version}</span>
                                </div>
                                {skill.description && (
                                    <div className="text-slate-500 text-[10px] line-clamp-2">{skill.description}</div>
                                )}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Sync Footer */}
            <div className="p-2 border-t border-slate-800 flex justify-end">
                <button onClick={fetchData} className="p-1 hover:bg-slate-800 rounded text-slate-500 hover:text-white" title="Refresh">
                    <RefreshCw size={12} />
                </button>
            </div>
        </div>
    );
};
