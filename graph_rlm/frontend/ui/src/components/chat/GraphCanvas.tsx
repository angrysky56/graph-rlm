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
        nodeColor={(node: any) => node.color || "#64748b"}
        nodeLabel="label"
        linkLabel={() => "Decomposes Into"}
        linkColor={() => "#475569"}
        backgroundColor="transparent"
        d3VelocityDecay={0.4} // Slightly higher friction for stability
        d3AlphaDecay={0.02}   // Slower cooling for better convergence
        nodeRelSize={6}
        nodeVal={(node: any) => node.val || 5} // Use calculated size
        // Interaction
        onNodeClick={(node) => onNodeClick && onNodeClick(node as GraphNode)}
        // Directional Arrows
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        // Particles for active nodes
        linkDirectionalParticles={(_link: any) => {
             // Optional: highlight links connected to processing nodes?
             // For now keep simple
             return 0;
        }}

      />

      {/* Overlay Status */}
      <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur border border-slate-700 p-2 rounded text-xs font-mono text-emerald-400 shadow-lg pointer-events-none select-none">
        GRAPH ENGINE: {data.nodes.length > 0 ? "ACTIVE" : "IDLE"}
        <div className="text-[10px] text-slate-500 mt-1">
          LOUVAIN CLUSTERING • BETWEENNESS CENTRALITY
        </div>
      </div>

      <div className="absolute bottom-4 right-4 text-[10px] text-slate-600 font-mono pointer-events-none select-none">
        NODES: {data.nodes.length} | EDGES: {data.links.length}
      </div>
    </div>
  );
};
