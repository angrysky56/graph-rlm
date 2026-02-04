from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import json
import os
import networkx as nx
from pathlib import Path

from ..logger import get_logger

logger = get_logger("graph_rlm.db.repository")

class GraphRepository(ABC):
    """
    Abstract interface for Graph Database operations.
    Standardizes output: methods returning nodes should return List[Dict] of properties.
    """

    @abstractmethod
    def create_thought_node(self, data: Dict[str, Any], parent_id: Optional[str] = None):
        pass

    @abstractmethod
    def get_parent_id(self, thought_id: str) -> Optional[str]:
        pass

    @abstractmethod
    def delete_thought_node(self, thought_id: str):
        pass

    @abstractmethod
    def update_thought_result(self, thought_id: str, data: Dict[str, Any]):
        pass

    @abstractmethod
    def get_graph_state(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_context_frontier(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_round(self, data: Dict[str, Any]):
        pass

    @abstractmethod
    def get_completed_rounds(self, root_session_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_session_trace(self, root_session_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_session(self, root_session_id: str):
        pass

    @abstractmethod
    def prune_orphans(self, cutoff_millis: int) -> int:
        pass

    @abstractmethod
    def reset_graph(self):
        pass

    @abstractmethod
    def mark_nodes_as_consolidated(self, node_ids: List[str], insight_id: str):
        pass

    @abstractmethod
    def perform_synaptic_homeostasis(self, cutoff_millis: int) -> int:
        pass

    @abstractmethod
    def get_context_scratchpad_data(self, root_session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_current_running_thought(self, root_session_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_session_thoughts(self, session_id: str, is_root: bool) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_thought(self, thought_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def create_insight(self, data: Dict[str, Any]):
        pass

    @abstractmethod
    def get_current_round_thoughts(self, root_session_id: str, current_round_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_sub_repls_data(self, root_session_id: str, current_session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_active_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_resolved_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        pass


class NetworkXRepository(GraphRepository):
    """
    In-memory Graph implementation using NetworkX.
    Persists to a JSON file (graph_db.json).
    """
    def __init__(self, persistence_path: str = "graph_db.json"):
        self.path = persistence_path
        self.graph = nx.DiGraph()
        self._load()

    def _save(self):
        data = nx.node_link_data(self.graph)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
            except Exception as e:
                logger.error(f"Failed to load graph DB: {e}")
                self.graph = nx.DiGraph()

    # ... Previous methods (shortened for clarity, assuming they are there or I just append/update)
    # I will paste the full content to be safe.

    def create_thought_node(self, data: Dict[str, Any], parent_id: Optional[str] = None):
        if "created_at" not in data:
            data["created_at"] = int(time.time() * 1000)
        self.graph.add_node(data["id"], **data, type="Thought")
        if parent_id and self.graph.has_node(parent_id):
            self.graph.add_edge(parent_id, data["id"], type="DECOMPOSES_INTO")
        self._save()

    def get_parent_id(self, thought_id: str) -> Optional[str]:
        if not self.graph.has_node(thought_id):
            return None
        for pred in self.graph.predecessors(thought_id):
            edge_data = self.graph.get_edge_data(pred, thought_id)
            if edge_data.get("type") == "DECOMPOSES_INTO":
                return pred
        return None

    def delete_thought_node(self, thought_id: str):
        if self.graph.has_node(thought_id):
            self.graph.remove_node(thought_id)
            self._save()

    def update_thought_result(self, thought_id: str, data: Dict[str, Any]):
        if self.graph.has_node(thought_id):
            if "completed_at" not in data:
                data["completed_at"] = int(time.time() * 1000)
            for k, v in data.items():
                self.graph.nodes[thought_id][k] = v
            self._save()

    def get_graph_state(self) -> List[Dict[str, Any]]:
        nodes = []
        for n, attr in self.graph.nodes(data=True):
            nodes.append(attr)
        return nodes

    def get_context_frontier(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        candidates = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought" and attr.get("session_id") == session_id:
                candidates.append(attr)
        candidates.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return candidates[:limit]

    def save_round(self, data: Dict[str, Any]):
        rid = data["round_id"]
        self.graph.add_node(rid, **data, type="Round")
        self._save()

    def get_completed_rounds(self, root_session_id: str) -> List[Dict[str, Any]]:
        rounds = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Round" and attr.get("root_session_id") == root_session_id:
                rounds.append(attr)
        rounds.sort(key=lambda x: x.get("ended_at", 0))
        return rounds

    def get_session_trace(self, root_session_id: str) -> List[Dict[str, Any]]:
        trace = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought" and attr.get("root_session_id") == root_session_id:
                trace.append(attr)
        trace.sort(key=lambda x: x.get("created_at", 0))
        return trace

    def delete_session(self, root_session_id: str):
        to_delete = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("root_session_id") == root_session_id:
                to_delete.append(n)
        for n in to_delete:
            self.graph.remove_node(n)
        self._save()

    def prune_orphans(self, cutoff_millis: int) -> int:
        to_delete = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought" and attr.get("created_at", 0) < cutoff_millis:
                if self.graph.degree(n) == 0:
                    to_delete.append(n)
        for n in to_delete:
            self.graph.remove_node(n)
        self._save()
        return len(to_delete)

    def reset_graph(self):
        self.graph.clear()
        self._save()

    def mark_nodes_as_consolidated(self, node_ids: List[str], insight_id: str):
        for nid in node_ids:
            if self.graph.has_node(nid):
                self.graph.nodes[nid]["status"] = "consolidated"
                self.graph.nodes[nid]["consolidated_at"] = int(time.time() * 1000)
                if self.graph.has_node(insight_id):
                    self.graph.add_edge(nid, insight_id, type="CONSOLIDATED_INTO")
        self._save()

    def perform_synaptic_homeostasis(self, cutoff_millis: int) -> int:
        to_delete = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("status") == "consolidated" and attr.get("consolidated_at", 0) < cutoff_millis:
                to_delete.append(n)
        for n in to_delete:
            self.graph.remove_node(n)
        self._save()
        return len(to_delete)

    def get_context_scratchpad_data(self, root_session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        sessions = {}
        for n, attr in self.graph.nodes(data=True):
             if attr.get("type") == "Thought" and (attr.get("root_session_id") == root_session_id or attr.get("session_id") == root_session_id):
                  sid = attr.get("session_id")
                  if sid not in sessions:
                       sessions[sid] = {"sid": sid, "count": 0, "prompts": [], "last_activity": 0}
                  sessions[sid]["count"] += 1
                  sessions[sid]["prompts"].append((attr.get("created_at", 0), attr.get("prompt", "")))
                  if attr.get("created_at", 0) > sessions[sid]["last_activity"]:
                       sessions[sid]["last_activity"] = attr.get("created_at", 0)

        results = []
        for sid, data in sessions.items():
             data["prompts"].sort(key=lambda x: x[0])
             initial_prompt = data["prompts"][0][1] if data["prompts"] else ""
             results.append({
                  "sid": sid,
                  "count": data["count"],
                  "prompt": initial_prompt,
                  "last_activity": data["last_activity"]
             })

        results.sort(key=lambda x: x["last_activity"], reverse=True)
        return results[:limit]

    def get_current_running_thought(self, root_session_id: str) -> Optional[Dict[str, Any]]:
         running = []
         for n, attr in self.graph.nodes(data=True):
              if attr.get("type") == "Thought" and attr.get("status") == "running":
                   if attr.get("root_session_id") == root_session_id or attr.get("session_id") == root_session_id:
                        running.append(attr)
         running.sort(key=lambda x: x.get("created_at", 0), reverse=True)
         return running[0] if running else None

    def get_session_thoughts(self, session_id: str, is_root: bool) -> List[Dict[str, Any]]:
         thoughts = []
         for n, attr in self.graph.nodes(data=True):
              if attr.get("type") == "Thought":
                   match = False
                   if is_root:
                        match = (attr.get("root_session_id") == session_id)
                   else:
                        match = (attr.get("session_id") == session_id)

                   if match:
                        thoughts.append(attr)
         thoughts.sort(key=lambda x: x.get("created_at", 0))
         return thoughts

    def get_thought(self, thought_id: str) -> Optional[Dict[str, Any]]:
        if self.graph.has_node(thought_id):
            return self.graph.nodes[thought_id]
        return None

    def create_insight(self, data: Dict[str, Any]):
        self.graph.add_node(data["id"], **data, type="Insight")
        self._save()

    def get_current_round_thoughts(self, root_session_id: str, current_round_id: str) -> List[Dict[str, Any]]:
        thoughts = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought" and attr.get("root_session_id") == root_session_id:
                # n.round_id IS NULL OR n.round_id = current_round_id
                rid = attr.get("round_id")
                if rid is None or rid == "" or rid == current_round_id:
                    thoughts.append(attr)
        thoughts.sort(key=lambda x: x.get("created_at", 0))
        return thoughts

    def get_sub_repls_data(self, root_session_id: str, current_session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        sessions = {}
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought" and attr.get("root_session_id") == root_session_id:
                sid = attr.get("session_id")
                if sid == current_session_id:
                    continue

                if sid not in sessions:
                    sessions[sid] = {"sid": sid, "prompts": [], "statuses": [], "results": [], "timestamps": []}

                sessions[sid]["prompts"].append(attr.get("prompt", ""))
                sessions[sid]["statuses"].append(attr.get("status", ""))
                sessions[sid]["results"].append(attr.get("result", ""))
                sessions[sid]["timestamps"].append(attr.get("created_at", 0))

        results = []
        for sid, data in sessions.items():
            # Logic: sort by timestamp, get last
            # We assume order of appending matches timestamp? No.
            # Let's zip and sort
            zipped = sorted(zip(data["timestamps"], data["prompts"], data["statuses"], data["results"]), key=lambda x: x[0])
            last = zipped[-1]
            first = zipped[0]

            results.append({
                "sid": sid,
                "initial_prompt": first[1],
                "last_activity": last[0],
                "last_status": last[2],
                "last_result": last[3],
                "last_action": last[1]
            })

        results.sort(key=lambda x: x["last_activity"], reverse=True)
        return results[:limit]

    def get_active_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        knots = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought":
                if attr.get("root_session_id") == root_session_id or attr.get("session_id") == session_id:
                    status = attr.get("status")
                    if status in ["failed", "error"]:
                        if not attr.get("dreamer_checked"):
                            knots.append(attr)
        knots.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return knots[:limit]

    def get_resolved_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        knots = []
        for n, attr in self.graph.nodes(data=True):
            if attr.get("type") == "Thought":
                if attr.get("root_session_id") == root_session_id or attr.get("session_id") == session_id:
                    status = attr.get("status")
                    checked = attr.get("dreamer_checked")
                    if status == "consolidated" or ((status in ["failed", "error"]) and checked):
                        knots.append(attr)
        knots.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return knots[:limit]


class FalkorDBRepository(GraphRepository):
    """
    Production implementation using FalkorDB.
    Normalizes outputs to List[Dict] to match NetworkX behavior.
    """
    def __init__(self, client):
        self.client = client

    def _unwrap(self, res: List[Dict[str, Any]], key: str = "n") -> List[Dict[str, Any]]:
        out = []
        for row in res:
            if key in row:
                val = row[key]
                if hasattr(val, "properties"):
                    out.append(val.properties)
                elif isinstance(val, dict):
                    out.append(val)
                else:
                    out.append(row)
            else:
                out.append(row)
        return out

    # ... (Existing methods)
    def create_thought_node(self, data: Dict[str, Any], parent_id: Optional[str] = None):
        tid = data["id"]
        cypher = "MERGE (t:Thought {id: $id}) SET t += $props"
        vec = None
        if "embedding" in data:
            vec = data.pop("embedding")
        self.client.query(cypher, {"id": tid, "props": data})
        if vec:
            self.client.query(f"MATCH (t:Thought {{id: $id}}) SET t.embedding = vecf32($vec)", {"id": tid, "vec": vec})
        if parent_id:
            self.client.query("MATCH (p:Thought {id: $pid}) MATCH (c:Thought {id: $tid}) MERGE (p)-[:DECOMPOSES_INTO]->(c)", {"pid": parent_id, "tid": tid})

    def get_parent_id(self, thought_id: str) -> Optional[str]:
        res = self.client.query("MATCH (p:Thought)-[:DECOMPOSES_INTO]->(c:Thought {id: $tid}) RETURN p.id as pid LIMIT 1", {"tid": thought_id})
        return res[0]["pid"] if res else None

    def delete_thought_node(self, thought_id: str):
        self.client.query("MATCH (n:Thought {id: $tid}) DETACH DELETE n", {"tid": thought_id})

    def update_thought_result(self, thought_id: str, data: Dict[str, Any]):
        cypher = "MATCH (t:Thought {id: $tid}) SET t += $props"
        vec = None
        if "embedding" in data:
            vec = data.pop("embedding")
        self.client.query(cypher, {"tid": thought_id, "props": data})
        if vec:
            self.client.query(f"MATCH (t:Thought {{id: $id}}) SET t.embedding = vecf32($vec)", {"id": thought_id, "vec": vec})

    def get_graph_state(self) -> List[Dict[str, Any]]:
        res = self.client.query("MATCH (n:Thought) RETURN n")
        return self._unwrap(res)

    def get_context_frontier(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        res = self.client.query(f"MATCH (n:Thought) WHERE n.session_id = $sid RETURN n ORDER BY n.created_at DESC LIMIT {limit}", {"sid": session_id})
        return self._unwrap(res)

    def save_round(self, data: Dict[str, Any]):
        self.client.query("CREATE (r:Round) SET r = $props", {"props": data})

    def get_completed_rounds(self, root_session_id: str) -> List[Dict[str, Any]]:
        res = self.client.query("MATCH (r:Round) WHERE r.root_session_id = $rsid RETURN r ORDER BY r.ended_at ASC", {"rsid": root_session_id})
        return self._unwrap(res, "r")

    def get_session_trace(self, root_session_id: str) -> List[Dict[str, Any]]:
        res = self.client.query("MATCH (n:Thought) WHERE n.root_session_id = $rsid RETURN n ORDER BY n.created_at ASC", {"rsid": root_session_id})
        return self._unwrap(res, "n")

    def delete_session(self, root_session_id: str):
        self.client.query("MATCH (n:Thought) WHERE n.root_session_id = $rsid DETACH DELETE n", {"rsid": root_session_id})
        self.client.query("MATCH (r:Round) WHERE r.root_session_id = $rsid DETACH DELETE r", {"rsid": root_session_id})

    def prune_orphans(self, cutoff_millis: int) -> int:
        res = self.client.query("MATCH (n:Thought) WHERE NOT (n)--() AND n.created_at < $cutoff DETACH DELETE n RETURN count(n) as count", {"cutoff": cutoff_millis})
        return res[0]["count"] if res else 0

    def reset_graph(self):
        self.client.query("MATCH (n) DETACH DELETE n")

    def mark_nodes_as_consolidated(self, node_ids: List[str], insight_id: str):
        self.client.query("""
            MATCH (t:Thought) WHERE t.id IN $ids
            SET t.status = 'consolidated', t.consolidated_at = timestamp()
            WITH t
            MATCH (i:Insight {id: $iid})
            MERGE (t)-[:CONSOLIDATED_INTO]->(i)
            """, {"ids": node_ids, "iid": insight_id})

    def perform_synaptic_homeostasis(self, cutoff_millis: int) -> int:
        res = self.client.query("MATCH (t:Thought) WHERE t.status = 'consolidated' AND t.consolidated_at < $cutoff DETACH DELETE t RETURN count(t) as count", {"cutoff": cutoff_millis})
        return res[0]["count"] if res else 0

    # ... (New methods)
    def get_context_scratchpad_data(self, root_session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $root_id OR n.session_id = $root_id)
            WITH n.session_id as sid, n, n.created_at as ts
            ORDER BY ts ASC
            WITH sid, count(n) as thought_count,
                 collect(n.prompt)[0] as initial_prompt,
                 max(ts) as last_activity
            RETURN sid, thought_count, initial_prompt, last_activity
            ORDER BY last_activity DESC
            LIMIT 20
        """
        res = self.client.query(q, {"root_id": root_session_id})
        out = []
        for row in res:
            if isinstance(row, dict):
                 out.append({
                      "sid": row.get("sid"),
                      "count": row.get("thought_count"),
                      "prompt": row.get("initial_prompt"),
                      "last_activity": row.get("last_activity")
                 })
            else:
                 out.append({
                      "sid": row[0],
                      "count": row[1],
                      "prompt": row[2],
                      "last_activity": row[3]
                 })
        return out

    def get_current_running_thought(self, root_session_id: str) -> Optional[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $root_id OR n.session_id = $root_id)
              AND n.status = 'running'
            RETURN n
            ORDER BY n.created_at DESC LIMIT 1
        """
        res = self.client.query(q, {"root_id": root_session_id})
        res = self._unwrap(res)
        return res[0] if res else None

    def get_session_thoughts(self, session_id: str, is_root: bool) -> List[Dict[str, Any]]:
        if is_root:
             q = "MATCH (n:Thought) WHERE n.root_session_id = $sid RETURN n ORDER BY n.created_at ASC"
        else:
             q = "MATCH (n:Thought) WHERE n.session_id = $sid RETURN n ORDER BY n.created_at ASC"
        res = self.client.query(q, {"sid": session_id})
        return self._unwrap(res)

    def get_thought(self, thought_id: str) -> Optional[Dict[str, Any]]:
        res = self.client.query("MATCH (n:Thought {id: $id}) RETURN n", {"id": thought_id})
        res = self._unwrap(res)
        return res[0] if res else None

    def create_insight(self, data: Dict[str, Any]):
        # data has id, content, created_at, type
        self.client.query("CREATE (i:Insight) SET i = $props", {"props": data})

    def get_current_round_thoughts(self, root_session_id: str, current_round_id: str) -> List[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $rsid
            AND (n.round_id IS NULL OR n.round_id = $crid)
            RETURN n
            ORDER BY n.created_at ASC
        """
        res = self.client.query(q, {"rsid": root_session_id, "crid": current_round_id})
        return self._unwrap(res)

    def get_sub_repls_data(self, root_session_id: str, current_session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id AND n.session_id <> $current_id
            WITH n.session_id as sid,
                 count(n) as thought_count,
                 collect(n.prompt)[0] as initial_prompt,
                 max(n.created_at) as last_activity,
                 collect(n.status)[-1] as last_status,
                 collect(n.result)[-1] as last_result,
                 collect(n.prompt)[-1] as last_action
            RETURN sid, thought_count, initial_prompt, last_activity, last_status, last_result, last_action
            ORDER BY last_activity DESC
            LIMIT 10
        """
        res = self.client.query(q, {"root_id": root_session_id, "current_id": current_session_id})
        out = []
        for row in res:
            if isinstance(row, dict):
                out.append({
                    "sid": row.get("sid"),
                    "initial_prompt": row.get("initial_prompt"),
                    "last_activity": row.get("last_activity"),
                    "last_status": row.get("last_status"),
                    "last_result": row.get("last_result"),
                    "last_action": row.get("last_action"),
                })
            else:
                out.append({
                    "sid": row[0],
                    "initial_prompt": row[2],
                    "last_activity": row[3],
                    "last_status": row[4],
                    "last_result": row[5],
                    "last_action": row[6],
                })
        return out

    def get_active_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (n.status = 'failed' OR n.status = 'error')
            AND (n.dreamer_checked IS NULL OR n.dreamer_checked = false)
            RETURN n
            ORDER BY n.created_at DESC LIMIT 3
        """
        res = self.client.query(q, {"rsid": root_session_id, "sid": session_id})
        return self._unwrap(res)

    def get_resolved_failure_knots(self, root_session_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $rsid OR n.session_id = $sid)
            AND (n.status = 'consolidated' OR ((n.status='failed' OR n.status='error') AND n.dreamer_checked = true))
            RETURN n
            ORDER BY n.created_at DESC LIMIT 3
        """
        res = self.client.query(q, {"rsid": root_session_id, "sid": session_id})
        return self._unwrap(res)
