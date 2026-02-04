import React, { useRef, useEffect, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useResizeObserver } from "../../hooks/useResizeObserver";
import { processGraphData } from "./graphProcessor";

// Graph Data Interface matching backend schema
interface GraphNode {
  id: string;
  label: string;
  group: number;
  val: number;
  status?: string;
}

interface GraphLink {
  source: string;
  target: string;
}

interface GraphCanvasProps {
  data: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  onNodeClick?: (node: GraphNode) => void;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({
  data,
  onNodeClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const dimensions = useResizeObserver(
    containerRef as React.RefObject<HTMLElement>,
  );
  const graphRef = useRef<any>(null);

  // Process data for aesthetics
  const processedData = useMemo(() => {
    return processGraphData(data.nodes, data.links);
  }, [data]); // Re-process when data object changes (includes node updates)

  // Auto-zoom when data changes significantly (optional, but nice)
  useEffect(() => {
    if (graphRef.current) {
      // Only zoom if node count is small (initial load) to avoid jarring jumps
      if (data.nodes.length < 5) {
        graphRef.current.zoomToFit(400);
      }
    }
  }, [data.nodes.length]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative overflow-hidden bg-slate-950"
    >
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        // dagMode="td" // REMOVED: Caused layout issues with cyclic/disconnected graphs
        // dagLevelDistance={50}
        nodeColor={(node: any) => node.color || "#64748b"}
        nodeLabel={(node: any) => {
          const promptSnippet = node.prompt
            ? node.prompt.length > 200
              ? node.prompt.substring(0, 200) + "..."
              : node.prompt
            : node.label;
          const resultSnippet = node.result
            ? `<div class="mt-1 pt-1 border-t border-slate-700 text-emerald-400 text-xs">${node.result.length > 100 ? node.result.substring(0, 100) + "..." : node.result}</div>`
            : "";

          return `
            <div class="bg-slate-900 border border-slate-700 p-2 rounded shadow-2xl max-w-xs font-sans">
              <div class="text-blue-400 font-bold text-[10px] uppercase tracking-tighter mb-1">${node.status || "Thought"}</div>
              <div class="text-slate-200 text-xs leading-tight">${promptSnippet}</div>
              ${resultSnippet}
              <div class="mt-2 text-[9px] text-slate-500 font-mono">ID: ${node.id.substring(0, 8)}...</div>
            </div>
          `;
        }}
        linkLabel={() => "Decomposes Into"}
        linkColor={() => "#475569"}
        backgroundColor="transparent"
        d3VelocityDecay={0.3} // Reduced friction to allow movement
        d3AlphaDecay={0.02} // Standard cooling
        nodeRelSize={7} // Highly visible nodes
        nodeVal={(node: any) => node.val || 5}
        warmupTicks={100} // Increase warmup
        // Interaction
        onNodeClick={(node) => onNodeClick && onNodeClick(node as GraphNode)}
        // Directional Arrows
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        // Simulation stability
        cooldownTicks={100}
        onEngineStop={() => {
           // Ensure we see the nodes when simulation stops
           if (graphRef.current) {
             graphRef.current.zoomToFit(400);
           }
        }}
      />

      {/* Overlay Status */}
      <div className="absolute bottom-4 right-4 text-[10px] text-slate-600 font-mono pointer-events-none select-none">
        NODES: {data.nodes.length} | EDGES: {data.links.length}
      </div>
    </div>
  );
};
