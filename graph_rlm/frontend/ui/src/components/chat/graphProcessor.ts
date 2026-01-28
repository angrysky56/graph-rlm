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

export const processGraphData = (
  rawNodes: any[],
  rawLinks: any[]
): { nodes: ProcessedNode[]; links: ProcessedLink[] } => {
  if (!rawNodes || rawNodes.length === 0) return { nodes: [], links: [] };

  try {
    const graph = new Graph();

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
          if (!graph.hasEdge(source, target)) {
               try {
                  graph.addEdge(source, target);
               } catch (e) {
                  // Ignore edge creation errors (e.g. self-loops if disallowed)
               }
          }
      }
    });

    // 2. Metrics: Betweenness Centrality
    let centralityScores: any = {};
    try {
        centralityScores = betweenness(graph);
    } catch (metricError) {
        console.warn("Graph metrics calculation failed:", metricError);
        // Fallback to empty scores
    }

    // 3. Communities: Louvain
    let communities: any = {};
    try {
        communities = louvain(graph);
    } catch (commError) {
        console.warn("Community detection failed:", commError);
        // Fallback
    }

    // 5. Reconstruct processed data
    const nodes = rawNodes.map((n) => {
      const commId = communities[n.id] ?? 0;
      const cent = centralityScores[n.id] ?? 0;

      const size = 4 + (cent * 20);

      return {
        ...n,
        community: commId,
        val: size,
        color: getNodeColor(commId),
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
