import React, { useState, useCallback } from 'react';
import { GraphCanvas } from '../chat/GraphCanvas';
import { InspectorPanel } from '../mcp/InspectorPanel';
import { Activity, ChevronRight, ChevronLeft, Maximize2, Minimize2, Cpu } from 'lucide-react';


// Add Scratchpad import (need to update imports first) and props
import { Scratchpad } from '../Scratchpad';

interface RightSidebarProps {
    graphData: { nodes: any[], links: any[] };
    onInjectContent: (text: string) => void;
    scratchpadText: string;
    onNodeSelect?: (node: any) => void;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({ graphData, onInjectContent, scratchpadText, onNodeSelect }) => {
    const handleNodeClick = useCallback((node: any) => {
        if (onNodeSelect) {
            onNodeSelect(node);
        }
    }, [onNodeSelect]);

    const [collapsed, setCollapsed] = useState(false);
    const [graphExpanded, setGraphExpanded] = useState(true);
    const [scratchpadExpanded, setScratchpadExpanded] = useState(true);
    const [toolsExpanded, setToolsExpanded] = useState(true);

    if (collapsed) {
        return (
            <div className="w-[40px] bg-slate-950 border-l border-slate-800 flex flex-col items-center py-4 gap-4">
                <button
                    onClick={() => setCollapsed(false)}
                    className="p-2 text-slate-500 hover:text-white transition-colors"
                >
                    <ChevronLeft size={16} />
                </button>
                <div className="h-[1px] w-6 bg-slate-800"></div>
                <div className="writing-vertical-rl text-[10px] font-bold tracking-widest text-slate-600 uppercase transform rotate-180 flex items-center gap-2 cursor-default">
                    <Activity size={10} /> KNOWLEDGE GRAPH
                </div>
            </div>
        );
    }

    return (
        <div className="w-[450px] bg-slate-950 border-l border-slate-800 flex flex-col h-screen transition-all shadow-2xl">
            {/* Main Header */}
            <div className="p-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/30">
                <span className="text-[10px] font-black text-slate-500 tracking-[0.2em] uppercase">CONTEXT & TOOLS</span>
                <button
                    onClick={() => setCollapsed(true)}
                    className="text-slate-600 hover:text-white transition-colors p-1"
                >
                    <ChevronRight size={14} />
                </button>
            </div>

            {/* Scratchpad Section (Primary) */}
            <div className={`flex flex-col border-b border-slate-800 transition-all duration-300 min-h-0 ${scratchpadExpanded ? 'flex-1 overflow-hidden' : 'h-[40px] flex-none'}`}>
                <div
                    className="flex items-center justify-between p-2 bg-slate-900/40 cursor-pointer hover:bg-slate-900/60 transition-colors"
                    onClick={() => setScratchpadExpanded(!scratchpadExpanded)}
                    title={scratchpadExpanded ? "Click to collapse" : "Click to expand"}
                >
                     <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-500 uppercase tracking-wider px-2">
                        <span>🧠 Agent Scratchpad <span className="text-slate-600 font-normal normal-case ml-1 font-mono">(Context)</span></span>
                    </div>
                     <div className="p-1 text-slate-600">
                        {scratchpadExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    </div>
                </div>
                {scratchpadExpanded && (
                     <div className="flex-1 overflow-hidden">
                        <Scratchpad scratchpadText={scratchpadText} />
                     </div>
                )}
            </div>

            {/* Live Graph Section */}
            <div className={`flex flex-col border-b border-slate-800 transition-all duration-300 min-h-0 ${graphExpanded ? 'flex-1 overflow-hidden' : 'h-[40px] flex-none'}`}>
                <div
                    className="flex items-center justify-between p-2 bg-slate-900/40 cursor-pointer hover:bg-slate-900/60 transition-colors"
                    onClick={() => setGraphExpanded(!graphExpanded)}
                    title={graphExpanded ? "Click to collapse" : "Click to expand"}
                >
                    <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">
                        <Activity size={12} className="text-blue-500" />
                        <span>Live Graph</span>
                        <span className="text-slate-700 font-mono">| {graphData.nodes.length} Nodes</span>
                    </div>
                    <div className="p-1 text-slate-600">
                        {graphExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    </div>
                </div>

                {graphExpanded && (
                    <div className="flex-1 relative overflow-hidden bg-slate-900/10">
                        <GraphCanvas
                            data={graphData}
                            onNodeClick={handleNodeClick}
                        />
                    </div>
                )}
            </div>

            {/* Tools & Skills (MCP Inspector) Section */}
            <div className={`flex flex-col transition-all duration-300 min-h-0 ${toolsExpanded ? 'flex-1 overflow-hidden' : 'h-[40px] flex-none'}`}>
                <div
                    className="flex items-center justify-between p-2 bg-slate-900/40 cursor-pointer hover:bg-slate-900/60 transition-colors"
                    onClick={() => setToolsExpanded(!toolsExpanded)}
                    title={toolsExpanded ? "Click to collapse" : "Click to expand"}
                >
                    <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">
                        <Cpu size={12} className="text-purple-500" />
                        <span>MCP Servers & Skills</span>
                    </div>
                    <div className="p-1 text-slate-600">
                        {toolsExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    </div>
                </div>
                {toolsExpanded && (
                    <div className="flex-1 flex flex-col min-h-0">
                        <InspectorPanel onInjectContent={onInjectContent} />
                    </div>
                )}
            </div>
        </div>
    );
};
