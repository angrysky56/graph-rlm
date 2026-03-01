import React, {
  useRef,
  useEffect,
  useMemo,
  useCallback,
  useState,
} from "react";
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
  type?: string;
}

interface GraphCanvasProps {
  data: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  onNodeClick?: (node: GraphNode) => void;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = React.memo(
  ({ data, onNodeClick }) => {
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
    }, [data.nodes, data.links]);

    // Apply forces for hierarchical clustering
    useEffect(() => {
      if (graphRef.current) {
        const fg = graphRef.current;
        fg.d3Force("charge").strength(-80);
        fg.d3Force("link").distance(35);
      }
    }, []);

    // Helper for link color
    const getLinkColor = useCallback((link: any) => {
      const type = link.type || "UNKNOWN";
      if (type === "CONTAINS" || type === "DECOMPOSES_INTO") return "#10b981"; // Emerald
      if (type === "RESONATES_WITH") return "#f59e0b"; // Amber (Semantic Similarity)
      return "#475569"; // Slate for generic or unknown
    }, []);

    // Link visibility/strength based on type
    const getLinkWidth = useCallback((link: any) => {
      const type = link.type || "UNKNOWN";
      if (type === "DECOMPOSES_INTO") return 2.5; // Thick: primary DAG structure
      if (type === "CONTAINS") return 1.5;
      return 0.8;
    }, []);

    const handleInternalClick = useCallback(
      (node: any) => {
        if (onNodeClick) {
          console.log("GraphCanvas: Internal Click Detected", node.id);
          onNodeClick(node as GraphNode);
        }
      },
      [onNodeClick],
    );

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setMousePos({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
        });
      }
    }, []);

    // Stable rendering function to prevent constant re-draws/resets
    const renderNode = useCallback(
      (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const r = node.val ? Math.sqrt(node.val) * 2.5 : 4;
        const label = node.label || node.id;

        // Overlay highlight if hovered
        const isHovered = hoveredNode && hoveredNode.id === node.id;
        if (isHovered) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, r + 2, 0, 2 * Math.PI, false);
          ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
          ctx.fill();
        }

        // Label rendering (only if zoomed in enough or hovered)
        if (globalScale > 1.2 || isHovered) {
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.6)";
          ctx.fillText(
            label.substring(0, 20),
            node.x,
            node.y + r + fontSize + 2,
          );
        }
      },
      [hoveredNode],
    );

    return (
      <div
        ref={containerRef}
        className="w-full h-full min-h-[400px] min-w-[300px] relative bg-slate-950"
        onMouseMove={handleMouseMove}
      >
        <ForceGraph2D
          ref={graphRef}
          width={dimensions.width || 450}
          height={dimensions.height || 400}
          graphData={processedData}
          nodeId="id"
          linkSource="source"
          linkTarget="target"
          nodeColor={(node: any) => node.color || "#3b82f6"}
          nodeRelSize={7}
          onNodeHover={setHoveredNode}
          onNodeClick={handleInternalClick}
          // Interaction
          enablePointerInteraction={true}
          enablePanInteraction={true}
          enableZoomInteraction={true}
          // Links — DAG-aware rendering
          linkColor={getLinkColor}
          linkWidth={getLinkWidth}
          linkDirectionalArrowLength={(l: any) => {
            const type = l.type || "UNKNOWN";
            // Larger arrows for structural DAG edges
            if (type === "DECOMPOSES_INTO") return 7;
            if (type === "CONTAINS") return 5;
            return 4;
          }}
          linkDirectionalArrowRelPos={1}
          // Flowing particles show causal direction on DAG edges
          linkDirectionalParticles={(l: any) => {
            const type = l.type || "UNKNOWN";
            if (type === "DECOMPOSES_INTO") return 3;
            if (type === "CONTAINS") return 2;
            return 0;
          }}
          linkDirectionalParticleSpeed={0.004}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleColor={getLinkColor}
          // Custom Layer: only for labels and highlights
          nodeCanvasObject={renderNode}
          nodeCanvasObjectMode="after"
          // Simulation — hierarchical DAG layout
          cooldownTicks={120}
          dagMode="td"
          dagLevelDistance={40}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
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
              opacity: hoveredNode ? 1 : 0,
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
                  <div>
                    SHEAF:{" "}
                    <span
                      className={
                        hoveredNode.sheaf_score > 0.6
                          ? "text-amber-400"
                          : "text-cyan-400"
                      }
                    >
                      {Number(hoveredNode.sheaf_score).toFixed(2)}
                    </span>
                  </div>
                  <div>
                    ENERGY:{" "}
                    <span className="text-violet-400">
                      {Number(hoveredNode.spectral_energy || 0).toFixed(2)}
                    </span>
                  </div>
                  {hoveredNode.h0_rank !== undefined && (
                    <div className="col-span-2">
                      H0 RANK:{" "}
                      <span className="text-pink-400">
                        {hoveredNode.h0_rank}
                      </span>
                    </div>
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
  },
);
