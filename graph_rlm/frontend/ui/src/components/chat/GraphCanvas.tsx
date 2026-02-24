import React, { useRef, useEffect, useMemo, useCallback, useState } from "react";
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

export const GraphCanvas: React.FC<GraphCanvasProps> = React.memo(({
  data,
  onNodeClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const dimensions = useResizeObserver(
    containerRef as React.RefObject<HTMLElement>,
  );
  const graphRef = useRef<any>(null);
  const [hoveredNode, setHoveredNode] = useState<any>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Process data for aesthetics
  const processedData = useMemo(() => {
    return processGraphData(data.nodes, data.links);
  }, [data.nodes.length, data.links.length]); // Re-process only when topology changes

  // Auto-zoom when data changes significantly (optional, but nice)
  useEffect(() => {
    if (graphRef.current && data.nodes.length > 0 && data.nodes.length < 5) {
        graphRef.current.zoomToFit(400);
    }
  }, [data.nodes.length]);

  const handleInternalClick = useCallback((node: any) => {
    if (onNodeClick) {
      console.log("GraphCanvas: Internal Click Detected", node.id);
      onNodeClick(node as GraphNode);
    }
  }, [onNodeClick]);

  const handleInternalHover = useCallback((node: any) => {
    if (node) {
      // console.log("GraphCanvas: Hovering", node.id);
    }
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setMousePos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-full h-full min-h-[400px] min-w-[300px] relative bg-slate-950 pointer-events-auto z-10 cursor-crosshair"
      onMouseMove={handleMouseMove}
    >
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width || 450}
        height={dimensions.height || 400}
        graphData={processedData}
        // dagMode="td" // REMOVED: Caused layout issues with cyclic/disconnected graphs
        // dagLevelDistance={50}
        nodeId="id"
        linkSource="source"
        linkTarget="target"
        nodeColor={(node: any) => node.color || "#94a3b8"} // Brighter default
        onNodeHover={(node: any) => {
          setHoveredNode(node);
          if (node) {
            (window as any).lastHoveredNode = node.id;
          }
          handleInternalHover(node);
        }}
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          ctx.fillStyle = color;
          const r = node.val ? Math.sqrt(node.val) * 5 : 20; // Increased pointer area
          ctx.beginPath(); ctx.arc(node.x, node.y, r + 5, 0, 2 * Math.PI, false); ctx.fill();
        }}
        linkLabel={() => "Relation"}
        linkColor={() => "#475569"}
        backgroundColor="transparent"
        nodeRelSize={9} // Increased relative size
        nodeVal={(node: any) => node.val || 8} // Increased base value
        // Interaction
        enablePointerInteraction={true}
        onNodeClick={handleInternalClick}
        // Arrows
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        // Simulation
        cooldownTicks={150}
        warmupTicks={100}
        d3VelocityDecay={0.4}
      />

      {/* Overlay Status */}
      <div className="absolute bottom-4 right-4 text-[10px] text-slate-600 font-mono pointer-events-none select-none">
        NODES: {data.nodes.length} | EDGES: {data.links.length}
      </div>

      {/* CUSTOM TOOLTIP */}
      {hoveredNode && (
        <div
          className="absolute z-[100] pointer-events-none transition-opacity duration-200"
          style={{
            left: mousePos.x + 15,
            top: mousePos.y + 15,
            opacity: hoveredNode ? 1 : 0
          }}
        >
          <div className="bg-slate-900/95 border border-slate-700 p-4 rounded-xl shadow-2xl backdrop-blur-md max-w-[350px]">
            <div className="text-[#60a5fa] font-bold text-[9px] uppercase tracking-wider mb-2">
              {String(hoveredNode.status || "Conceptual Unit").toUpperCase()}
            </div>
            <div className="text-slate-100 text-sm leading-snug font-medium mb-2">
              {hoveredNode.prompt || hoveredNode.label || "Abstract Thought"}
            </div>
            {hoveredNode.result && (
              <div className="mt-3 pt-3 border-t border-slate-800 text-emerald-400 text-xs leading-relaxed italic">
                {String(hoveredNode.result)}
              </div>
            )}
            {hoveredNode.sheaf_score !== undefined && (
              <div className="mt-4 pt-3 border-t border-slate-800 grid grid-cols-2 gap-3 font-mono text-[10px] text-slate-400">
                <div>SHEAF: <span className={hoveredNode.sheaf_score > 0.6 ? "text-amber-400" : "text-cyan-400"}>{Number(hoveredNode.sheaf_score).toFixed(2)}</span></div>
                <div>ENERGY: <span className="text-violet-400">{Number(hoveredNode.spectral_energy || 0).toFixed(2)}</span></div>
                {hoveredNode.h0_rank !== undefined && (
                  <div className="col-span-2">H0 RANK: <span className="text-pink-400">{hoveredNode.h0_rank}</span></div>
                )}
              </div>
            )}
            <div className="mt-3 pt-2 border-t border-dashed border-slate-800 text-[8px] text-slate-600 font-mono">
              REF: {String(hoveredNode.id).substring(0, 12)}...
            </div>
          </div>
        </div>
      )}
    </div>
  );
});
