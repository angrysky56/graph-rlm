import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import { betweenness } from "graphology-metrics/centrality";

export interface ProcessedNode {
  id: string;
  label: string;
  group: number; // Original group (status-based)
  val: number; // Size (based on centrality)
  status?: string;
  community: number; // Louvain community
  color: string; // Community color
  x?: number;
  y?: number;
  sheaf_score?: number;
  spectral_energy?: number;
  h0_rank?: number;
}

export interface ProcessedLink {
  source: string;
  target: string;
}

// Infranodus-inspired vibrant palette for dark mode
const PALETTE = [
  "#3b82f6", // Blue
  "#a855f7", // Purple
  "#10b981", // Green
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#ec4899", // Pink
  "#06b6d4", // Cyan
  "#8b5cf6", // Violet
  "#f97316", // Orange
  "#14b8a6", // Teal
];

const getNodeColor = (communityId: number): string => {
  return PALETTE[communityId % PALETTE.length];
};

const getSemanticColor = (node: any, defaultColor: string): string => {
  if (node.status === 'LOGICAL_KNOT') return '#ef4444'; // Red
  if (node.status === 'SEMANTIC_DRIFT') return '#f59e0b'; // Amber
  if (node.status === 'error') return '#dc2626'; // Dark Red
  if (node.sheaf_score && node.sheaf_score > 0.5) return '#f97316'; // Orange (High Energy)
  return defaultColor;
};

export const processGraphData = (
  rawNodes: any[],
  rawLinks: any[]
): { nodes: ProcessedNode[]; links: ProcessedLink[] } => {
  if (!rawNodes || rawNodes.length === 0) return { nodes: [], links: [] };

  try {
    // Explicitly enforce undirected graph for Louvain/Betweenness to avoid "mixed graph" errors
    const graph = new Graph({ type: 'undirected', allowSelfLoops: false });

    // 1. Build Graph
    rawNodes.forEach((n) => {
      // Ensure node is valid
      if (n && n.id && !graph.hasNode(n.id)) {
        // Create a defensive copy
        graph.addNode(n.id, { ...n });
      }
    });

    rawLinks.forEach((l) => {
      // Ensure source/target exist and handle object references (react-force-graph mutation)
      const source = typeof l.source === 'object' ? l.source.id : l.source;
      const target = typeof l.target === 'object' ? l.target.id : l.target;

      if (source && target && graph.hasNode(source) && graph.hasNode(target)) {
          // In an undirected graph, addEdge is order-independent
          if (!graph.hasEdge(source, target)) {
               try {
                  graph.addEdge(source, target);
               } catch (e) {
                  // Ignore edge creation errors
               }
          }
      }
    });

    // 2. Metrics: Betweenness Centrality (Skipped for performance if graph is large)
    let centralityScores: any = {};
    const isLargeGraph = rawNodes.length > 500;

    if (!isLargeGraph) {
        try {
            centralityScores = betweenness(graph);
        } catch (metricError) {
            console.warn("Graph metrics calculation failed:", metricError);
        }
    }

    // 3. Communities: Louvain (Skipped if large)
    let communities: any = {};
    if (!isLargeGraph) {
        try {
            communities = louvain(graph);
        } catch (commError) {
            console.warn("Community detection failed:", commError);
        }
    }

    // 5. Reconstruct processed data
    const nodes = rawNodes.map((n) => {
      const commId = communities[n.id] ?? 0;
      const cent = centralityScores[n.id] ?? 0;

      const size = 4 + (cent * 20);

      return {
        ...n,
        // Preserve existing coordinates if they exist
        x: n.x,
        y: n.y,
        vx: n.vx,
        vy: n.vy,
        community: commId,
        val: size,
        color: getSemanticColor(n, getNodeColor(commId)),
        sheaf_score: n.sheaf_score,
        spectral_energy: n.spectral_energy,
        h0_rank: n.h0_rank,
      };
    });

    return { nodes, links: rawLinks };

  } catch (error) {
    console.error("Graph processing failed:", error);
    // Fallback: return raw data with defaults
    const fallbackNodes = rawNodes.map(n => ({
        ...n,
        val: 5,
        color: "#64748b",
        community: 0
    }));
    return { nodes: fallbackNodes, links: rawLinks };
  }
};
