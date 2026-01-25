import React, { useState } from 'react';
import { GraphCanvas } from '../chat/GraphCanvas';
import { InspectorPanel } from '../mcp/InspectorPanel';
import { Activity, ChevronRight, ChevronLeft, Maximize2, Minimize2 } from 'lucide-react';


interface RightSidebarProps {
    graphData: { nodes: any[], links: any[] };
    onInjectContent: (text: string) => void;
}

export const RightSidebar: React.FC<RightSidebarProps> = ({ graphData, onInjectContent }) => {
    const [collapsed, setCollapsed] = useState(false);
    const [graphExpanded, setGraphExpanded] = useState(true);

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
        <div className="w-[450px] bg-slate-950 border-l border-slate-800 flex flex-col h-screen transition-all">
            {/* Header */}
            <div className="p-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/30">
                <span className="text-xs font-bold text-slate-400">CONTEXT & TOOLS</span>
                <button
                    onClick={() => setCollapsed(true)}
                    className="text-slate-600 hover:text-white transition-colors"
                >
                    <ChevronRight size={14} />
                </button>
            </div>

            {/* Graph Section */}
            <div className={`flex flex-col border-b border-slate-800 transition-all duration-300 ${graphExpanded ? 'h-[400px]' : 'h-[40px]'}`}>
                <div className="flex items-center justify-between p-2 bg-slate-900/20">
                    <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">
                        <Activity size={12} /> Live Graph
                        <span className="text-slate-700">| {graphData.nodes.length} Nodes</span>
                    </div>
                    <button
                        onClick={() => setGraphExpanded(!graphExpanded)}
                        className="p-1 text-slate-600 hover:text-white"
                    >
                        {graphExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                    </button>
                </div>

                {graphExpanded && (
                    <div className="flex-1 relative overflow-hidden bg-slate-900/10">
                        <GraphCanvas data={graphData} />
                    </div>
                )}
            </div>

            {/* MCP Inspector Section */}
            <div className="flex-1 flex flex-col min-h-0">
                <InspectorPanel onInjectContent={onInjectContent} />
            </div>
        </div>
    );
};
