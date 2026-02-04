import React, { useState } from "react";
import { GraphCanvas } from "../chat/GraphCanvas";
import { SessionList } from "../layout/SessionList";
import { MessageSquare, PanelLeftClose, PanelLeftOpen } from "lucide-react";

interface GraphExplorerProps {
  graphData: { nodes: any[]; links: any[] };
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onSwitchToChat: () => void;
  onNodeClick: (node: any) => void;
  onShowFullGraph: () => void;
}

export const GraphExplorer: React.FC<GraphExplorerProps> = ({
  graphData,
  activeSessionId,
  onSelectSession,
  onSwitchToChat,
  onNodeClick,
  onShowFullGraph,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="w-full h-screen bg-slate-950 flex overflow-hidden">
      {/* 1. Integrated Sidebar (Session History) */}
      <div
        className={`
            relative z-20 h-full bg-slate-900/95 backdrop-blur border-r border-slate-700
            transition-all duration-300 ease-in-out flex flex-col shrink-0
            ${sidebarOpen ? "w-80 translate-x-0" : "w-0 -translate-x-full opacity-0"}
        `}
      >
        <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-black/20 shrink-0">
          <div>
            <h2 className="text-emerald-400 font-bold text-sm tracking-wider">
              KNOWLEDGE GRAPH
            </h2>
            <div className="text-[10px] text-slate-500 font-mono mt-1">
              Topological Analysis
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden p-2">
            <SessionList
                activeSessionId={activeSessionId}
                onSelectSession={onSelectSession}
            />
        </div>

        {/* Full Graph Toggle */}
        <div className="p-3 border-t border-slate-800 bg-slate-900/50">
             <button
                onClick={onShowFullGraph}
                className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-wider rounded border border-slate-700 transition-colors flex items-center justify-center gap-2"
             >
                <span>🌍 Show Full Graph</span>
             </button>
        </div>
      </div>

      {/* Sidebar Toggle Button */}
      <div className={`absolute top-4 z-30 transition-all duration-300 ${sidebarOpen ? "left-80 ml-4" : "left-4"}`}>
           <button
             onClick={() => setSidebarOpen(!sidebarOpen)}
             className="p-2 rounded bg-slate-800 text-slate-400 hover:text-white border border-slate-700 shadow-xl"
             title={sidebarOpen ? "Collapse List" : "Show Sessions"}
           >
             {sidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
           </button>
      </div>


      {/* 2. Main Canvas Area */}
      <div className="flex-1 relative flex flex-col min-w-0 bg-slate-950">
        {/* Graph Visualizer - Direct Flex Child */}
        <div className="flex-1 relative overflow-hidden">
          <GraphCanvas data={graphData} onNodeClick={onNodeClick} />
        </div>

        {/* Top Right Controls */}
        <div className="absolute top-4 right-4 z-30 flex gap-2">
          {/* View Switcher: Go to Chat */}
          <button
            onClick={onSwitchToChat}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded shadow-lg border border-blue-400/50 transition-all font-bold text-xs uppercase tracking-wider"
          >
            <MessageSquare size={14} />
            <span>Open Chat</span>
          </button>
        </div>

        {/* Overlay Info (Bottom Left) */}
        {!sidebarOpen && (
            <div className="absolute bottom-4 left-4 z-30 pointer-events-none">
                <div className="bg-black/50 backdrop-blur p-2 rounded text-slate-500 text-[10px] font-mono">
                    SESSION: {activeSessionId || "GLOBAL VIEW"}
                </div>
            </div>
        )}
      </div>
    </div>
  );
};
