import React, { useRef, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { useResizeObserver } from "../../hooks/useResizeObserver";

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

  // Auto-zoom when data changes significantly (optional, but nice)
  useEffect(() => {
    if (graphRef.current) {
      // Gentle fit
      // graphRef.current.zoomToFit(400);
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
        graphData={data}
        nodeColor={(node: any) => {
          switch (node.group) {
            case 1:
              return "#3b82f6"; // Blue (Root)
            case 2:
              return "#a855f7"; // Purple (Processing)
            case 3:
              return "#10b981"; // Green (Completed)
            case 4:
              return "#f59e0b"; // Amber (Verification)
            default:
              return "#64748b"; // Slate
          }
        }}
        nodeLabel="label"
        linkLabel={() => "Decomposes Into"}
        linkColor={() => "#475569"}
        backgroundColor="transparent"
        d3VelocityDecay={0.3}
        nodeRelSize={6}
        // Interaction
        onNodeClick={(node) => onNodeClick && onNodeClick(node as GraphNode)}
        // Directional Arrows
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
      />

      {/* Overlay Status */}
      <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur border border-slate-700 p-2 rounded text-xs font-mono text-emerald-400 shadow-lg">
        GRAPH ENGINE: {data.nodes.length > 0 ? "ACTIVE" : "IDLE"}
      </div>

      <div className="absolute bottom-4 right-4 text-[10px] text-slate-600 font-mono">
        NODES: {data.nodes.length} | EDGES: {data.links.length}
      </div>
    </div>
  );
};
