"""Personal Knowledge Graph, Episodic-to-Semantic Consolidation & Contextual Session Engine.

Engineered for persistent user memory, personal knowledge graphs (PKG), temporal contradiction
supersession, Ebbinghaus retention modeling, 2-hop hybrid graph+BM25 contextual session
assembly, and user-controlled memory governance (Pin / Forget / Restore).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

TOKEN_RE = re.compile(r"[a-z0-9_]+")
MEM_CITE_RE = re.compile(r"\[(MEM-\d+)\]")
EDGE_CITE_RE = re.compile(r"\[(EDGE-\d+)\]")

ENTITY_TYPES = ("User", "Person", "Project", "Metric", "System", "Preference", "Commitment")
EDGE_TYPES = (
    "OWNS",
    "COLLABORATES_WITH",
    "BLOCKED_BY",
    "PREFERS_FORMAT",
    "SUPERSEDES",
    "TRACKS_METRIC",
    "COMMITTED_TO",
)


def compute_ebbinghaus_retention(
    age_days: float,
    access_count: int,
    lambda_rate: float = 0.045,
    pinned: bool = False,
    forgotten: bool = False,
) -> float:
    """Compute Ebbinghaus memory retention boosted by retrieval/access frequency.

    R(t, n) = exp(-lambda * t / (1 + 0.48 * ln(1 + n)))
    Pinned memories lock at 1.0; forgotten/tombstoned memories drop to 0.0.
    """
    if forgotten:
        return 0.0
    if pinned:
        return 1.0
    effective_age = max(0.0, float(age_days))
    freq_boost = 1.0 + 0.48 * math.log1p(max(0, int(access_count)))
    score = math.exp((-float(lambda_rate) * effective_age) / freq_boost)
    return round(min(1.0, max(0.02, score)), 4)


def retention_curve_points(
    access_count: int,
    lambda_rate: float = 0.045,
    pinned: bool = False,
    forgotten: bool = False,
) -> List[Dict[str, float]]:
    """Sample 60-day Ebbinghaus retention trajectory for UI plotting."""
    days = [0, 3, 7, 14, 21, 30, 45, 60]
    return [
        {
            "day": d,
            "retention": round(
                compute_ebbinghaus_retention(
                    age_days=float(d),
                    access_count=access_count,
                    lambda_rate=lambda_rate,
                    pinned=pinned,
                    forgotten=forgotten,
                )
                * 100.0,
                1,
            ),
        }
        for d in days
    ]


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _char_trigrams(text: str) -> Dict[str, int]:
    cleaned = re.sub(r"\s+", " ", (text or "").lower().strip())
    if len(cleaned) < 3:
        return {cleaned: 1} if cleaned else {}
    counts: Dict[str, int] = {}
    for idx in range(len(cleaned) - 2):
        tri = cleaned[idx : idx + 3]
        counts[tri] = counts.get(tri, 0) + 1
    return counts


def _cosine_trigrams(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _approx_tokens(text: str) -> int:
    return max(1, int(round(len((text or "").split()) * 1.32)))


@dataclass
class MemoryNode:
    memory_id: str
    entity_type: str  # User | Person | Project | Metric | System | Preference | Commitment
    label: str
    summary: str
    slot_key: Optional[str] = None
    status: str = "ACTIVE"  # ACTIVE | SUPERSEDED | TOMBSTONED
    age_days: float = 1.0
    access_count: int = 3
    lambda_rate: float = 0.045
    pinned: bool = False
    forgotten: bool = False
    provenance_sessions: List[str] = field(default_factory=list)
    evidence_quote: str = ""
    superseded_by: Optional[str] = None
    created_session: str = "SESS-01"

    def retention_score(self) -> float:
        return compute_ebbinghaus_retention(
            age_days=self.age_days,
            access_count=self.access_count,
            lambda_rate=self.lambda_rate,
            pinned=self.pinned,
            forgotten=self.forgotten,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        ret = self.retention_score()
        payload["retention_score"] = ret
        payload["retention_pct"] = round(ret * 100.0, 1)
        payload["retention_curve"] = retention_curve_points(
            access_count=self.access_count,
            lambda_rate=self.lambda_rate,
            pinned=self.pinned,
            forgotten=self.forgotten,
        )
        return payload


@dataclass
class MemoryEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str  # OWNS | COLLABORATES_WITH | BLOCKED_BY | PREFERS_FORMAT | SUPERSEDES | TRACKS_METRIC | COMMITTED_TO
    slot_key: Optional[str] = None
    status: str = "ACTIVE"  # ACTIVE | SUPERSEDED | PRUNED
    session_id: str = "SESS-01"
    confidence: float = 0.96
    superseded_by_edge: Optional[str] = None
    superseded_in_session: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodicSession:
    session_id: str
    title: str
    timestamp_label: str
    age_days: float
    transcript: str
    extracted_memory_ids: List[str] = field(default_factory=list)
    extracted_edge_ids: List[str] = field(default_factory=list)
    superseded_edge_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["raw_tokens"] = _approx_tokens(self.transcript)
        return d


class PersonaKnowledgeGraph:
    """Single user's Personal Knowledge Graph with episodic-to-semantic consolidation."""

    def __init__(
        self,
        persona_id: str,
        user_name: str,
        role_title: str,
        org_unit: str,
        root_summary: str,
    ) -> None:
        self.persona_id = persona_id
        self.user_name = user_name
        self.role_title = role_title
        self.org_unit = org_unit
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.sessions: List[EpisodicSession] = []
        self._mem_seq = 0
        self._edge_seq = 0
        self._sess_seq = 0

        self.user_node_id = "USER-00"
        self.nodes[self.user_node_id] = MemoryNode(
            memory_id=self.user_node_id,
            entity_type="User",
            label=user_name,
            summary=root_summary,
            status="ACTIVE",
            age_days=0.0,
            access_count=25,
            pinned=True,
            provenance_sessions=["SESS-01"],
            evidence_quote=root_summary,
        )

    def _next_mem_id(self) -> str:
        self._mem_seq += 1
        return f"MEM-{self._mem_seq:02d}"

    def _next_edge_id(self) -> str:
        self._edge_seq += 1
        return f"EDGE-{self._edge_seq:02d}"

    def _next_sess_id(self) -> str:
        self._sess_seq += 1
        return f"SESS-{self._sess_seq:02d}"

    def find_node_by_label(self, label: str, entity_type: Optional[str] = None) -> Optional[MemoryNode]:
        norm = label.strip().lower()
        for node in self.nodes.values():
            if node.memory_id == self.user_node_id:
                continue
            if node.label.strip().lower() == norm:
                if entity_type is None or node.entity_type == entity_type:
                    return node
        return None

    def upsert_entity_with_relation(
        self,
        *,
        entity_type: str,
        label: str,
        summary: str,
        relation: str,
        source_id: Optional[str] = None,
        slot_key: Optional[str] = None,
        session_id: str,
        age_days: float = 0.2,
        access_count: int = 4,
        pinned: bool = False,
        evidence_quote: str = "",
        contradicts_label: Optional[str] = None,
    ) -> Tuple[MemoryNode, MemoryEdge, List[MemoryEdge]]:
        """Insert or reinforce an entity + directed edge, resolving any slot or explicit contradiction."""
        src = source_id or self.user_node_id
        existing = self.find_node_by_label(label, entity_type=entity_type)
        if existing:
            node = existing
            node.summary = summary or node.summary
            node.age_days = min(node.age_days, age_days)
            node.access_count += 2
            if node.forgotten:
                node.forgotten = False
            node.status = "ACTIVE"
            if session_id not in node.provenance_sessions:
                node.provenance_sessions.append(session_id)
            if evidence_quote:
                node.evidence_quote = evidence_quote
            if slot_key:
                node.slot_key = slot_key
        else:
            mem_id = self._next_mem_id()
            node = MemoryNode(
                memory_id=mem_id,
                entity_type=entity_type,
                label=label.strip(),
                summary=summary.strip(),
                slot_key=slot_key,
                status="ACTIVE",
                age_days=age_days,
                access_count=access_count,
                pinned=pinned,
                forgotten=False,
                provenance_sessions=[session_id],
                evidence_quote=evidence_quote.strip() or summary.strip(),
                created_session=session_id,
            )
            self.nodes[mem_id] = node

        superseded_edges: List[MemoryEdge] = []
        prior_target_ids: List[str] = []

        # Detect prior active edges that conflict on (source_id, relation, slot_key) or explicit contradicts_label
        for edge in list(self.edges.values()):
            if edge.status != "ACTIVE":
                continue
            if edge.target_id == node.memory_id and edge.source_id == src and edge.relation == relation:
                continue
            same_slot = bool(
                slot_key
                and edge.source_id == src
                and edge.relation == relation
                and edge.slot_key == slot_key
            )
            explicit_match = False
            if contradicts_label and edge.source_id == src:
                tgt_node = self.nodes.get(edge.target_id)
                if tgt_node and contradicts_label.strip().lower() in tgt_node.label.lower():
                    explicit_match = True
            if same_slot or explicit_match:
                superseded_edges.append(edge)
                prior_target_ids.append(edge.target_id)

        # Check if identical active edge already exists
        active_existing_edge: Optional[MemoryEdge] = None
        for edge in self.edges.values():
            if (
                edge.source_id == src
                and edge.target_id == node.memory_id
                and edge.relation == relation
                and edge.status == "ACTIVE"
            ):
                active_existing_edge = edge
                break

        if active_existing_edge is not None:
            new_edge = active_existing_edge
            new_edge.session_id = session_id
        else:
            edge_id = self._next_edge_id()
            new_edge = MemoryEdge(
                edge_id=edge_id,
                source_id=src,
                target_id=node.memory_id,
                relation=relation,
                slot_key=slot_key,
                status="ACTIVE",
                session_id=session_id,
                confidence=0.97,
                reason=f"Consolidated from {session_id}",
            )
            self.edges[edge_id] = new_edge

        # Mark conflicting old edges & nodes as SUPERSEDED and emit explicit SUPERSEDES lineage edge
        for old_edge in superseded_edges:
            old_edge.status = "SUPERSEDED"
            old_edge.superseded_by_edge = new_edge.edge_id
            old_edge.superseded_in_session = session_id
            old_node = self.nodes.get(old_edge.target_id)
            if old_node and old_node.memory_id != self.user_node_id:
                old_node.status = "SUPERSEDED"
                old_node.superseded_by = node.memory_id

        for old_tid in prior_target_ids:
            if old_tid != node.memory_id:
                sup_edge_id = self._next_edge_id()
                self.edges[sup_edge_id] = MemoryEdge(
                    edge_id=sup_edge_id,
                    source_id=node.memory_id,
                    target_id=old_tid,
                    relation="SUPERSEDES",
                    slot_key=slot_key,
                    status="ACTIVE",
                    session_id=session_id,
                    confidence=0.99,
                    reason=f"Temporal contradiction resolved in {session_id}: {node.label} supersedes {self.nodes[old_tid].label}",
                )

        return node, new_edge, superseded_edges

    def ingest_structured_session(
        self,
        *,
        title: str,
        timestamp_label: str,
        age_days: float,
        transcript: str,
        facts: List[Dict[str, Any]],
    ) -> EpisodicSession:
        sess_id = self._next_sess_id()
        mem_ids: List[str] = []
        edge_ids: List[str] = []
        sup_ids: List[str] = []

        label_to_id: Dict[str, str] = {}
        for f in facts:
            src_id = self.user_node_id
            if f.get("source_label"):
                src_match = label_to_id.get(f["source_label"].lower())
                if not src_match:
                    found_node = self.find_node_by_label(f["source_label"])
                    if found_node:
                        src_match = found_node.memory_id
                if src_match:
                    src_id = src_match

            node, edge, sup_edges = self.upsert_entity_with_relation(
                entity_type=f["entity_type"],
                label=f["label"],
                summary=f["summary"],
                relation=f["relation"],
                source_id=src_id,
                slot_key=f.get("slot_key"),
                session_id=sess_id,
                age_days=age_days,
                access_count=int(f.get("access_count", 4)),
                pinned=bool(f.get("pinned", False)),
                evidence_quote=f.get("evidence_quote", ""),
                contradicts_label=f.get("contradicts_label"),
            )
            label_to_id[node.label.lower()] = node.memory_id
            if node.memory_id not in mem_ids:
                mem_ids.append(node.memory_id)
            if edge.edge_id not in edge_ids:
                edge_ids.append(edge.edge_id)
            for se in sup_edges:
                if se.edge_id not in sup_ids:
                    sup_ids.append(se.edge_id)

        session_obj = EpisodicSession(
            session_id=sess_id,
            title=title,
            timestamp_label=timestamp_label,
            age_days=age_days,
            transcript=transcript,
            extracted_memory_ids=mem_ids,
            extracted_edge_ids=edge_ids,
            superseded_edge_ids=sup_ids,
        )
        self.sessions.append(session_obj)
        return session_obj

    def ingest_live_transcript(self, session_title: str, transcript: str) -> Dict[str, Any]:
        """Parse a live work session note/transcript into typed PKG nodes, edges & contradiction supersessions."""
        cleaned = (transcript or "").strip()
        if not cleaned:
            raise ValueError("Session transcript cannot be empty.")

        facts: List[Dict[str, Any]] = []
        sentences = [
            s.strip()
            for s in re.split(r"[\n\.!?]+", cleaned)
            if s.strip() and len(s.strip()) > 4
        ]

        current_active_project_label: Optional[str] = None
        for e in self.edges.values():
            if e.source_id == self.user_node_id and e.relation == "OWNS" and e.status == "ACTIVE":
                n = self.nodes.get(e.target_id)
                if n and n.entity_type == "Project":
                    current_active_project_label = n.label

        for sent in sentences:
            s_clean = re.sub(r"^\[[0-9:]+\]\s*[^:]{1,40}:\s*", "", sent).strip()

            # 1. Ownership switch / new project ownership
            m_switch_proj = re.search(
                r"(?:switched|transitioned|moved|reassigned|pivoted)\s+(?:project\s+|ownership\s+)?from\s+([A-Za-z0-9_\-\s/]{2,42}?)\s+to\s+([A-Za-z0-9_\-\s/]{2,45})(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_switch_proj and not re.search(r"sql|dialect|redshift|athena|trino|snowflake|spark|java|python|rust|format|memo|bluf", s_clean, re.I):
                old_proj = m_switch_proj.group(1).strip(" .,-")
                new_proj = m_switch_proj.group(2).strip(" .,-")
                current_active_project_label = new_proj
                facts.append(
                    {
                        "entity_type": "Project",
                        "label": new_proj,
                        "summary": f"Active owned initiative ({new_proj}); superseded prior ownership of {old_proj}.",
                        "relation": "OWNS",
                        "slot_key": "primary_project",
                        "contradicts_label": old_proj,
                        "access_count": 6,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            m_now_own = re.search(
                r"(?:now\s+owning|now\s+own|taking\s+over\s+ownership\s+of|primary\s+project\s+is\s+now|leading\s+initiative)\s+([A-Za-z0-9_\-\s/]{3,46}?)(?:\s+instead\s+of\s+([A-Za-z0-9_\-\s/]{2,40}))?(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_now_own:
                new_proj = m_now_own.group(1).strip(" .,-")
                old_proj = (m_now_own.group(2) or "").strip(" .,-") or None
                current_active_project_label = new_proj
                facts.append(
                    {
                        "entity_type": "Project",
                        "label": new_proj,
                        "summary": f"Current primary project owned by {self.user_name}: {new_proj}.",
                        "relation": "OWNS",
                        "slot_key": "primary_project",
                        "contradicts_label": old_proj,
                        "access_count": 6,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            # 2. SQL / Warehouse / Stack / Format preference switch
            m_switch_pref = re.search(
                r"(?:switched|migrated|changed|updated|moving)\s+(?:preferred\s+)?([A-Za-z0-9_\-\s]{0,28}?)\s*from\s+([A-Za-z0-9_\-\+\s/]{2,40}?)\s+to\s+([A-Za-z0-9_\-\+\s/]{2,48})(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_switch_pref:
                domain_hint = (m_switch_pref.group(1) or "").lower()
                old_pref = m_switch_pref.group(2).strip(" .,-")
                new_pref = m_switch_pref.group(3).strip(" .,-")
                combined_hint = f"{domain_hint} {old_pref.lower()} {new_pref.lower()}"
                if any(k in combined_hint for k in ("sql", "redshift", "athena", "iceberg", "trino", "snowflake", "bigquery", "duckdb")):
                    slot = "sql_dialect"
                elif any(k in combined_hint for k in ("java", "python", "rust", "go", "typescript", "stack", "runtime")):
                    slot = "code_stack"
                else:
                    slot = "doc_format"
                facts.append(
                    {
                        "entity_type": "Preference",
                        "label": new_pref,
                        "summary": f"Preferred {slot.replace('_', ' ')}: {new_pref} (replaces {old_pref}).",
                        "relation": "PREFERS_FORMAT",
                        "slot_key": slot,
                        "contradicts_label": old_pref,
                        "access_count": 7,
                        "pinned": True,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            m_pref = re.search(
                r"(?:prefer|prefers|standardize\s+on|always\s+format\s+as|use\s+dialect|switched\s+to\s+using)\s+([A-Za-z0-9_\-\+\s/:]{3,54}?)(?:\s+(?:over|instead\s+of|rather\s+than)\s+([A-Za-z0-9_\-\+\s/]{2,40}))?(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_pref:
                new_pref = m_pref.group(1).strip(" .,-")
                old_pref = (m_pref.group(2) or "").strip(" .,-") or None
                low = f"{new_pref.lower()} {(old_pref or '').lower()} {s_clean.lower()}"
                if any(k in low for k in ("sql", "athena", "iceberg", "redshift", "trino", "snowflake", "duckdb")):
                    slot = "sql_dialect"
                elif any(k in low for k in ("python", "rust", "java", "typescript", "golang", "stack")):
                    slot = "code_stack"
                elif any(k in low for k in ("bluf", "bullet", "memo", "pr/faq", "table", "diff", "format", "summary", "brief")):
                    slot = "doc_format"
                else:
                    slot = "workflow_preference"
                facts.append(
                    {
                        "entity_type": "Preference",
                        "label": new_pref,
                        "summary": f"Active user preference ({slot}): {new_pref}.",
                        "relation": "PREFERS_FORMAT",
                        "slot_key": slot,
                        "contradicts_label": old_pref,
                        "access_count": 6,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            # 3. Blocker / System dependency
            m_block = re.search(
                r"(?:blocked\s+by|bottlenecked\s+on|gated\s+on|waiting\s+on)\s+([A-Za-z0-9_\-\s/()]{4,58})(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_block:
                blocker_label = m_block.group(1).strip(" .,-")
                facts.append(
                    {
                        "entity_type": "System",
                        "label": blocker_label,
                        "summary": f"Active engineering/launch blocker: {blocker_label}.",
                        "relation": "BLOCKED_BY",
                        "source_label": current_active_project_label,
                        "slot_key": "active_blocker",
                        "access_count": 5,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            # 4. Stakeholder / Collaborator
            m_collab = re.search(
                r"(?i:collaborating\s+(?:closely\s+)?with|partnering\s+(?:closely\s+)?with|working\s+(?:closely\s+)?with|syncing\s+with|aligned\s+with)\s+([A-Z][A-Za-z\.\-\s]{2,42}(?:\([A-Za-z0-9\s,&\-/]+\))?)",
                s_clean,
            )
            if m_collab:
                person_lbl = m_collab.group(1).strip(" .,-")
                facts.append(
                    {
                        "entity_type": "Person",
                        "label": person_lbl,
                        "summary": f"Cross-functional stakeholder collaborating with {self.user_name}: {person_lbl}.",
                        "relation": "COLLABORATES_WITH",
                        "access_count": 5,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            # 5. Metric / SLA target
            m_metric = re.search(
                r"(?:target\s+metric|tracking\s+metric|sla\s+target|kpi\s+is|goal\s+is)\s*[:\-]?\s*([A-Za-z0-9_\-\s<=>$%\.]{4,55})",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_metric:
                metric_lbl = m_metric.group(1).strip(" .,-")
                facts.append(
                    {
                        "entity_type": "Metric",
                        "label": metric_lbl,
                        "summary": f"Tracked operational KPI / SLA target: {metric_lbl}.",
                        "relation": "TRACKS_METRIC",
                        "source_label": current_active_project_label,
                        "access_count": 5,
                        "evidence_quote": s_clean,
                    }
                )
                continue

            # 6. Commitment / Cadence
            m_commit = re.search(
                r"(?:committed\s+to|committing\s+to|shipping|delivering|deadline\s+is)\s+([A-Za-z0-9_\-\s:@/]{5,58})(?:$|[,;])",
                s_clean,
                flags=re.IGNORECASE,
            )
            if m_commit:
                comm_lbl = m_commit.group(1).strip(" .,-")
                facts.append(
                    {
                        "entity_type": "Commitment",
                        "label": comm_lbl,
                        "summary": f"Active delivery commitment: {comm_lbl}.",
                        "relation": "COMMITTED_TO",
                        "access_count": 5,
                        "evidence_quote": s_clean,
                    }
                )
                continue

        # Fallback if freeform note had no explicit pattern match: extract concise semantic commitment/note
        if not facts:
            short_lbl = " ".join(cleaned.split()[:6]).strip(".,;:-")
            facts.append(
                {
                    "entity_type": "Commitment",
                    "label": short_lbl or session_title[:42],
                    "summary": cleaned[:180],
                    "relation": "COMMITTED_TO",
                    "access_count": 4,
                    "evidence_quote": cleaned[:180],
                }
            )

        session_obj = self.ingest_structured_session(
            title=session_title.strip() or "Ad-Hoc Contextual Work Session",
            timestamp_label="Just now (Live Session)",
            age_days=0.05,
            transcript=cleaned,
            facts=facts,
        )

        return {
            "session": session_obj.to_dict(),
            "extracted_nodes": [self.nodes[mid].to_dict() for mid in session_obj.extracted_memory_ids],
            "extracted_edges": [self.edges[eid].to_dict() for eid in session_obj.extracted_edge_ids],
            "superseded_edges": [self.edges[eid].to_dict() for eid in session_obj.superseded_edge_ids],
        }

    def toggle_memory_governance(self, memory_id: str, action: str) -> Dict[str, Any]:
        """Pin, unpin, forget (tombstone + edge prune), or restore a personal memory entity."""
        if memory_id not in self.nodes:
            raise KeyError(f"Unknown memory_id: {memory_id}")
        if memory_id == self.user_node_id:
            raise ValueError("Cannot tombstone root User identity node.")

        node = self.nodes[memory_id]
        act = (action or "").strip().lower()
        affected_edges: List[str] = []

        if act == "pin":
            node.pinned = not node.pinned
            if node.pinned and node.forgotten:
                node.forgotten = False
                node.status = "ACTIVE"
        elif act == "forget":
            node.forgotten = True
            node.pinned = False
            node.status = "TOMBSTONED"
            for edge in self.edges.values():
                if (edge.source_id == memory_id or edge.target_id == memory_id) and edge.status == "ACTIVE":
                    if edge.relation != "SUPERSEDES":
                        edge.status = "PRUNED"
                        affected_edges.append(edge.edge_id)
        elif act == "restore":
            node.forgotten = False
            node.status = "SUPERSEDED" if node.superseded_by else "ACTIVE"
            for edge in self.edges.values():
                if (edge.source_id == memory_id or edge.target_id == memory_id) and edge.status == "PRUNED":
                    other_id = edge.target_id if edge.source_id == memory_id else edge.source_id
                    other_node = self.nodes.get(other_id)
                    if other_node and not other_node.forgotten:
                        edge.status = "SUPERSEDED" if edge.superseded_by_edge else "ACTIVE"
                        affected_edges.append(edge.edge_id)
        else:
            raise ValueError(f"Unsupported governance action '{action}'. Expected pin | forget | restore.")

        return {
            "memory": node.to_dict(),
            "action": act,
            "affected_edge_ids": affected_edges,
        }

    def retrieve_2hop_context(self, prompt: str, top_k_seeds: int = 4) -> Dict[str, Any]:
        """Hybrid BM25 + Trigram Cosine + Ebbinghaus Retention + 2-Hop PKG Neighborhood Retriever."""
        t0 = time.perf_counter()
        q_tokens = _tokenize(prompt)
        q_tris = _char_trigrams(prompt)

        active_candidates: List[MemoryNode] = []
        suppressed_contradictions: List[Dict[str, Any]] = []
        tombstoned_excluded: List[str] = []

        # Document frequencies for BM25
        doc_tokens_map: Dict[str, List[str]] = {}
        df_counts: Dict[str, int] = {}
        for mid, node in self.nodes.items():
            if mid == self.user_node_id:
                continue
            text = f"{node.label} {node.summary} {node.entity_type} {node.slot_key or ''} {node.evidence_quote}"
            toks = _tokenize(text)
            doc_tokens_map[mid] = toks
            for ut in set(toks):
                df_counts[ut] = df_counts.get(ut, 0) + 1

        total_docs = max(1, len(doc_tokens_map))
        avgdl = max(1.0, sum(len(v) for v in doc_tokens_map.values()) / total_docs)

        scored_active: List[Tuple[float, MemoryNode]] = []
        for mid, node in self.nodes.items():
            if mid == self.user_node_id:
                continue
            toks = doc_tokens_map[mid]
            dl = max(1, len(toks))
            tf_map: Dict[str, int] = {}
            for t in toks:
                tf_map[t] = tf_map.get(t, 0) + 1

            bm25 = 0.0
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf > 0:
                    df = df_counts.get(qt, 1)
                    idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                    bm25 += idf * ((tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * (dl / avgdl))))

            sem = _cosine_trigrams(q_tris, _char_trigrams(f"{node.label} {node.summary}"))
            lexical_semantic = bm25 * 0.68 + sem * 2.4

            if node.forgotten or node.status == "TOMBSTONED":
                if lexical_semantic > 0.25:
                    tombstoned_excluded.append(f"{node.memory_id} ({node.label})")
                continue

            if node.status == "SUPERSEDED":
                replacement = self.nodes.get(node.superseded_by or "")
                suppressed_contradictions.append(
                    {
                        "stale_memory_id": node.memory_id,
                        "stale_label": node.label,
                        "entity_type": node.entity_type,
                        "superseded_by_id": replacement.memory_id if replacement else node.superseded_by,
                        "superseded_by_label": replacement.label if replacement else "Newer Fact",
                        "would_have_matched_query": round(lexical_semantic, 3) > 0.15,
                    }
                )
                continue

            active_candidates.append(node)
            ret = node.retention_score()
            pin_bonus = 0.45 if node.pinned else 0.0
            pref_bonus = 0.38 if node.entity_type in ("Preference", "Commitment") else 0.0
            proj_bonus = 0.30 if node.entity_type == "Project" else 0.0
            final_score = (lexical_semantic + pref_bonus + proj_bonus + pin_bonus) * (0.55 + 0.45 * ret)
            scored_active.append((final_score, node))

        scored_active.sort(key=lambda pair: pair[0], reverse=True)
        seed_nodes = [n for _, n in scored_active[:top_k_seeds]]

        # Always include active Preferences & Pinned items in persistent session envelope
        hop1_ids: Set[str] = {n.memory_id for n in seed_nodes}
        for n in active_candidates:
            if n.pinned or n.entity_type == "Preference":
                hop1_ids.add(n.memory_id)

        # Expand 2-hop neighborhood across ACTIVE directed edges (excluding SUPERSEDES pointers to stale nodes)
        hop2_ids: Set[str] = set()
        selected_edges: List[MemoryEdge] = []
        seen_edge_ids: Set[str] = set()

        # First hop from user_node_id and seed nodes
        frontier = set(hop1_ids) | {self.user_node_id}
        for edge in self.edges.values():
            if edge.status != "ACTIVE" or edge.relation == "SUPERSEDES":
                continue
            s_node = self.nodes.get(edge.source_id)
            t_node = self.nodes.get(edge.target_id)
            if not s_node or not t_node or s_node.forgotten or t_node.forgotten:
                continue
            if s_node.status != "ACTIVE" or t_node.status != "ACTIVE":
                continue

            if edge.source_id in frontier and edge.target_id in hop1_ids:
                if edge.edge_id not in seen_edge_ids:
                    selected_edges.append(edge)
                    seen_edge_ids.add(edge.edge_id)
            elif edge.source_id in hop1_ids or edge.target_id in hop1_ids:
                other = edge.target_id if edge.source_id in hop1_ids else edge.source_id
                if other != self.user_node_id and other not in hop1_ids:
                    hop2_ids.add(other)
                if edge.edge_id not in seen_edge_ids:
                    selected_edges.append(edge)
                    seen_edge_ids.add(edge.edge_id)

        selected_node_ids = list(hop1_ids | hop2_ids)
        selected_nodes = [
            self.nodes[mid]
            for mid in sorted(selected_node_ids)
            if mid in self.nodes and self.nodes[mid].status == "ACTIVE" and not self.nodes[mid].forgotten
        ]

        # Reinforce Ebbinghaus access counts for retrieved active memories
        for n in selected_nodes:
            n.access_count += 1

        # Assemble compact PKG triple context block
        context_lines = [
            f"USER: {self.user_name} ({self.role_title})",
            "ACTIVE_PKG_NODES:",
        ]
        for n in selected_nodes:
            context_lines.append(
                f"[{n.memory_id}] {n.entity_type}: {n.label} | {n.summary}"
            )
        context_lines.append("ACTIVE_PKG_EDGES:")
        for e in selected_edges:
            context_lines.append(
                f"[{e.edge_id}] {e.source_id} -{e.relation}-> {e.target_id}"
            )

        assembled_context = "\n".join(context_lines)
        pkg_prompt_tokens = _approx_tokens(assembled_context) + _approx_tokens(prompt)
        full_history_text = "\n\n".join(
            f"[{s.session_id} | {s.timestamp_label} | {s.title}]\n{s.transcript}"
            for s in self.sessions
        )
        full_history_tokens = _approx_tokens(full_history_text) + _approx_tokens(prompt)
        token_reduction_pct = round(
            (1.0 - (pkg_prompt_tokens / float(max(1, full_history_tokens)))) * 100.0, 1
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return {
            "assembled_context": assembled_context,
            "hop1_nodes": [self.nodes[mid].to_dict() for mid in sorted(hop1_ids) if mid in self.nodes],
            "hop2_nodes": [self.nodes[mid].to_dict() for mid in sorted(hop2_ids) if mid in self.nodes],
            "retrieved_nodes": [n.to_dict() for n in selected_nodes],
            "retrieved_edges": [e.to_dict() for e in selected_edges],
            "suppressed_contradictions": suppressed_contradictions,
            "tombstoned_excluded": tombstoned_excluded,
            "retrieval_latency_ms": elapsed_ms,
            "pkg_prompt_tokens": pkg_prompt_tokens,
            "full_history_tokens": full_history_tokens,
            "token_reduction_pct": token_reduction_pct,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        all_nodes = [n.to_dict() for n in self.nodes.values()]
        active_edges = [e.to_dict() for e in self.edges.values() if e.status == "ACTIVE"]
        superseded_edges = [e.to_dict() for e in self.edges.values() if e.status != "ACTIVE"]
        preferences = [
            n.to_dict()
            for n in self.nodes.values()
            if n.entity_type in ("Preference", "Commitment") and n.memory_id != self.user_node_id
        ]
        return {
            "persona_id": self.persona_id,
            "user_name": self.user_name,
            "role_title": self.role_title,
            "org_unit": self.org_unit,
            "user_node_id": self.user_node_id,
            "nodes": all_nodes,
            "active_edges": active_edges,
            "superseded_edges": superseded_edges,
            "all_edges": [e.to_dict() for e in self.edges.values()],
            "preferences": preferences,
            "sessions": [s.to_dict() for s in self.sessions],
            "stats": {
                "active_nodes": sum(1 for n in self.nodes.values() if n.status == "ACTIVE" and not n.forgotten),
                "superseded_nodes": sum(1 for n in self.nodes.values() if n.status == "SUPERSEDED"),
                "tombstoned_nodes": sum(1 for n in self.nodes.values() if n.forgotten),
                "active_edges": len(active_edges),
                "superseded_edges": len(superseded_edges),
                "sessions_consolidated": len(self.sessions),
            },
        }


def _invoke_bedrock_nova_single(system_prompt: str, user_prompt: str, max_tokens: int = 380) -> Optional[Dict[str, Any]]:
    """Invoke Amazon Bedrock (`us.amazon.nova-lite-v1:0`) via boto3 or local AWS CLI."""
    if os.environ.get("QUICKMEMORY_OFFLINE_LLM") == "1":
        return None

    model_id = os.environ.get("QUICKMEMORY_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
    region = os.environ.get("AWS_REGION", "us-west-2")
    t0 = time.perf_counter()

    # Path A: boto3 in AWS Lambda Python 3.12 runtime or local venv
    try:
        import boto3  # type: ignore

        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.15},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return {
            "text": text.strip(),
            "engine": f"aws-bedrock ({model_id})",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tokens": resp.get("usage", {}).get("totalTokens", 0),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Path B: Local AWS CLI fallback
    aws_bin = shutil.which("aws")
    if not aws_bin:
        return None
    profile = os.environ.get("QUICKMEMORY_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = profile

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": max_tokens, "temperature": 0.15}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": f"aws-bedrock ({model_id})",
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "tokens": data.get("usage", {}).get("totalTokens", 0),
            }
    except Exception:
        pass
    return None


def _build_deterministic_stateless_response(prompt: str) -> str:
    q = prompt.lower()
    if "sql" in q or "query" in q or "latency" in q or "wbr" in q or "weekly" in q:
        return (
            "Here is a generic status template and standard ANSI/PostgreSQL query you can adapt:\n\n"
            "1. Project Overview: Summarize your team's current initiative, key milestones, and owners.\n"
            "2. Sample SQL (Generic Warehouse):\n"
            "   SELECT service_name, AVG(response_time_ms) AS avg_latency\n"
            "   FROM service_logs\n"
            "   WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'\n"
            "   GROUP BY service_name;\n\n"
            "Please share which project you own, your preferred SQL warehouse dialect (e.g., Redshift, Snowflake, or BigQuery), "
            "and your preferred executive readout format so I can tailor this."
        )
    return (
        "I don't have context from your prior work sessions yet. To give you a tailored answer, please let me know:\n"
        "- Which service or initiative you currently own\n"
        "- Key stakeholders, active blockers, and SLA targets\n"
        "- Your preferred output structure, language/SQL stack, and review cadence."
    )


def _build_deterministic_personalized_response(
    pkg: PersonaKnowledgeGraph,
    prompt: str,
    retrieval: Dict[str, Any],
) -> str:
    nodes_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for n in retrieval["retrieved_nodes"]:
        nodes_by_type.setdefault(n["entity_type"], []).append(n)

    edges = retrieval["retrieved_edges"]
    edge_lookup: Dict[str, str] = {e["target_id"]: e["edge_id"] for e in edges}

    proj = (nodes_by_type.get("Project") or [None])[0]
    prefs = nodes_by_type.get("Preference") or []
    metrics = nodes_by_type.get("Metric") or []
    systems = nodes_by_type.get("System") or []
    people = nodes_by_type.get("Person") or []
    commits = nodes_by_type.get("Commitment") or []

    lines: List[str] = []
    if proj:
        e_tag = f" [{edge_lookup[proj['memory_id']]}]" if proj["memory_id"] in edge_lookup else ""
        lines.append(
            f"• **BLUF — Active Initiative (`{proj['label']}`)**: Scoped directly to **{proj['label']}** [{proj['memory_id']}]{e_tag} "
            f"(stale superseded initiatives automatically excluded)."
        )
    if metrics or systems:
        parts = []
        if metrics:
            m0 = metrics[0]
            parts.append(f"tracking **{m0['label']}** [{m0['memory_id']}]")
        if systems:
            s0 = systems[0]
            se_tag = f" [{edge_lookup[s0['memory_id']]}]" if s0["memory_id"] in edge_lookup else ""
            parts.append(f"unblocking **{s0['label']}** [{s0['memory_id']}]{se_tag}")
        lines.append(f"• **Operational Target & Critical Path**: Currently {' and '.join(parts)}.")
    if people or commits:
        c_parts = []
        if people:
            p0 = people[0]
            pe_tag = f" [{edge_lookup[p0['memory_id']]}]" if p0["memory_id"] in edge_lookup else ""
            c_parts.append(f"aligned with **{p0['label']}** [{p0['memory_id']}]{pe_tag}")
        if commits:
            cm0 = commits[0]
            c_parts.append(f"for **{cm0['label']}** [{cm0['memory_id']}]")
        lines.append(f"• **Stakeholder & Cadence Alignment**: Prepared {' '.join(c_parts)}.")

    pref_citations = " ".join(f"[{p['memory_id']}]" for p in prefs[:2])
    pref_labels = ", ".join(p["label"] for p in prefs[:2])
    if prefs:
        lines.append(
            f"\n**Applied Persistent Preferences ({pref_labels})** {pref_citations}:"
        )

    sql_pref = next((p for p in prefs if "athena" in p["label"].lower() or "iceberg" in p["label"].lower() or "sql" in (p.get("slot_key") or "")), None)
    if sql_pref and ("sql" in prompt.lower() or "query" in prompt.lower() or "wbr" in prompt.lower() or "latency" in prompt.lower()):
        lines.append(
            "```sql\n"
            f"-- Dialect: {sql_pref['label']} [{sql_pref['memory_id']}] (Redshift SQL superseded)\n"
            "SELECT checkout_region,\n"
            "       approx_percentile(checkout_latency_ms, 0.99) AS p99_latency_ms,\n"
            "       COUNT(*) FILTER (WHERE checkout_latency_ms > 180) AS sla_breaches\n"
            "FROM quick_lakehouse.checkout_telemetry_iceberg\n"
            "WHERE event_ts >= current_timestamp - INTERVAL '7' DAY\n"
            "GROUP BY 1 ORDER BY p99_latency_ms DESC;\n"
            "```"
        )
    return "\n".join(lines)


class QuickMemoryEngine:
    """Multi-Persona Persistent Memory & Personal Knowledge Graph Engine."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.personas: Dict[str, PersonaKnowledgeGraph] = {}
        self.reset()

    @staticmethod
    def load_measured_benchmarks() -> Dict[str, Any]:
        bench_path = Path(__file__).resolve().parent.parent / "benchmarks" / "latest_results.json"
        if bench_path.is_file():
            try:
                return json.loads(bench_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"status": "awaiting_benchmark_run", "comparison_rows": []}

    def reset(self) -> None:
        with self._lock:
            self.personas = {
                "principal_pm": self._seed_principal_pm(),
                "staff_sde": self._seed_staff_sde(),
            }

    def _seed_principal_pm(self) -> PersonaKnowledgeGraph:
        pkg = PersonaKnowledgeGraph(
            persona_id="principal_pm",
            user_name="Maya Chen",
            role_title="Principal Product Manager — Checkout & FinOps",
            org_unit="Amazon Quick Enterprise",
            root_summary="Principal PM driving Q4 checkout latency reduction and weekly executive WBR analytics.",
        )

        # Session 1 (35 days ago) — Initial Q3 baseline (contains legacy facts superseded later)
        pkg.ingest_structured_session(
            title="Session 01 · Q3 Settlement Kickoff & Redshift Reporting Setup",
            timestamp_label="35 days ago",
            age_days=35.0,
            transcript=(
                "[00:01] Maya Chen: Team, kicking off our Q3 commerce review. Right now I am owning Payments-v1 "
                "across NA and EU merchant settlement batches, reconciling gateway retry queues and authorization holds. "
                "[03:18] Priya Nair (FinOps Lead): For margin attribution on Payments-v1, we pull nightly settlement tables "
                "from the legacy Redshift cluster (prod_finops_rs.settlement_ledger) and join gateway interchange tiers. "
                "[06:42] Maya Chen: Agreed. Whenever Quick drafts analytics queries for my workspace, please write all "
                "queries in Redshift Warehouse SQL (DATEADD, APPROXIMATE PERCENTILE_DISC, Redshift DISTKEY conventions) "
                "and format leadership updates as a 6-Page Narrative Memo with full prose background sections and appendix tables. "
                "Partnering closely with Priya Nair (FinOps Lead) on interchange cost-per-order attribution through Q3 close."
            ),
            facts=[
                {
                    "entity_type": "Project",
                    "label": "Payments-v1",
                    "summary": "Legacy Q3 payment settlement pipeline previously owned by Maya Chen.",
                    "relation": "OWNS",
                    "slot_key": "primary_project",
                    "access_count": 2,
                    "evidence_quote": "Right now I am owning Payments-v1 across NA and EU merchant settlement batches.",
                },
                {
                    "entity_type": "Preference",
                    "label": "Redshift Warehouse SQL",
                    "summary": "Legacy Q3 preference for Amazon Redshift SQL warehouse syntax (DATEADD, PERCENTILE_DISC).",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "sql_dialect",
                    "access_count": 2,
                    "evidence_quote": "Please write all queries in Redshift Warehouse SQL.",
                },
                {
                    "entity_type": "Preference",
                    "label": "6-Page Narrative Memo",
                    "summary": "Initial Q3 preference for long-form 6-page prose narratives.",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "doc_format",
                    "access_count": 1,
                    "evidence_quote": "Format leadership updates as a 6-Page Narrative Memo.",
                },
                {
                    "entity_type": "Person",
                    "label": "Priya Nair (FinOps Lead)",
                    "summary": "Finance operations partner owning checkout cost-per-transaction attribution.",
                    "relation": "COLLABORATES_WITH",
                    "access_count": 5,
                    "evidence_quote": "Partnering closely with Priya Nair (FinOps Lead) on interchange cost-per-order attribution.",
                },
            ],
        )

        # Session 2 (29 days ago) — Legacy Redshift WLM tuning & 6-page memo review
        pkg.ingest_structured_session(
            title="Session 02 · Q3 Payments-v1 Merchant Settlement Retro & Redshift WLM",
            timestamp_label="29 days ago",
            age_days=29.0,
            transcript=(
                "[00:00] Maya Chen: Reviewing week-9 merchant settlement batch SLAs on Payments-v1. Yesterday's EU SEPA "
                "clearing batch queued for 18 minutes inside prod_finops_rs Redshift WLM slot group 2 while our nightly "
                "Redshift Warehouse SQL DATEADD cohort rollups executed. "
                "[04:15] Priya Nair (FinOps Lead): I appended the interchange fee variance breakdown into section 4 of "
                "our 6-Page Narrative Memo for Q3 closeout. Let's keep monitoring Payments-v1 authorization hold expirations "
                "until Core Ledger Operations staffs the Q4 handoff."
            ),
            facts=[],
        )

        # Session 3 (24 days ago) — Historical Payments-v1 closeout & Redshift freeze notice
        pkg.ingest_structured_session(
            title="Session 03 · Q3 Payments-v1 Handoff Prep & Redshift Freeze Notice",
            timestamp_label="24 days ago",
            age_days=24.0,
            transcript=(
                "[00:00] Maya Chen: Wrapping up final Q3 merchant reconciliation on Payments-v1 before handoff to the "
                "Core Ledger Operations team. Our legacy Redshift Warehouse SQL scripts on prod_finops_rs ran in 14 minutes "
                "during month-end close, and Data Platform announced that prod_finops_rs is entering read-only freeze next week. "
                "[04:55] Priya Nair (FinOps Lead): Once Payments-v1 transitions to maintenance mode, FinOps will mirror "
                "cost allocation tags into the new S3 Apache Iceberg lakehouse catalog so Q4 latency initiatives can join "
                "unit economics without hitting Redshift WLM queues."
            ),
            facts=[],
        )

        # Session 4 (11 days ago) — Ownership pivot + SQL warehouse migration (Temporal Contradictions!)
        pkg.ingest_structured_session(
            title="Session 04 · Q4 Ownership Pivot to Checkout-Latency-Q4 & Athena Iceberg Cutover",
            timestamp_label="11 days ago",
            age_days=11.0,
            transcript=(
                "[00:02] Maya Chen: Important Q4 charter change — I formally handed off Payments-v1 to Core Ledger Ops "
                "and switched ownership to Checkout-Latency-Q4 as my sole flagship program through Cyber Week. "
                "Do not reference Payments-v1 as my active project anymore. "
                "[03:10] Maya Chen: Second, our telemetry warehouse completed migration off Redshift to Apache Iceberg on S3. "
                "Standardize on Athena Iceberg SQL (Trino approx_percentile, current_timestamp - INTERVAL '7' DAY, "
                "quick_lakehouse.checkout_telemetry_iceberg) instead of Redshift Warehouse SQL for every query you write. "
                "[07:25] Maya Chen: Our north-star KPI for Checkout-Latency-Q4 is p99 Checkout Latency <= 180ms "
                "(down from 242ms baseline), and Checkout-Latency-Q4 is currently blocked by LakeFormation Cross-Account Grants "
                "on the EU checkout Iceberg partitions."
            ),
            facts=[
                {
                    "entity_type": "Project",
                    "label": "Checkout-Latency-Q4",
                    "summary": "Q4 flagship program cutting global one-click checkout tail latency under 180ms.",
                    "relation": "OWNS",
                    "slot_key": "primary_project",
                    "contradicts_label": "Payments-v1",
                    "access_count": 11,
                    "pinned": True,
                    "evidence_quote": "I formally handed off Payments-v1 and switched ownership to Checkout-Latency-Q4.",
                },
                {
                    "entity_type": "Preference",
                    "label": "Athena Iceberg SQL",
                    "summary": "Always write analytical SQL using AWS Athena / Trino Iceberg dialect (approx_percentile, INTERVAL '7' DAY).",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "sql_dialect",
                    "contradicts_label": "Redshift Warehouse SQL",
                    "access_count": 9,
                    "pinned": True,
                    "evidence_quote": "Standardize on Athena Iceberg SQL instead of Redshift Warehouse SQL.",
                },
                {
                    "entity_type": "Metric",
                    "label": "p99 Checkout Latency <= 180ms",
                    "summary": "Hard Q4 SLA target: reduce global checkout p99 from 242ms down to <= 180ms before Cyber Week.",
                    "relation": "TRACKS_METRIC",
                    "source_label": "Checkout-Latency-Q4",
                    "access_count": 8,
                    "evidence_quote": "Our north-star KPI for Checkout-Latency-Q4 is p99 Checkout Latency <= 180ms.",
                },
                {
                    "entity_type": "System",
                    "label": "LakeFormation Cross-Account Grants",
                    "summary": "IAM + AWS LakeFormation permission bottleneck blocking EU checkout Iceberg table joins.",
                    "relation": "BLOCKED_BY",
                    "source_label": "Checkout-Latency-Q4",
                    "slot_key": "active_blocker",
                    "access_count": 6,
                    "evidence_quote": "Checkout-Latency-Q4 is currently blocked by LakeFormation Cross-Account Grants.",
                },
            ],
        )

        # Session 5 (7 days ago) — Checkout token cache & Iceberg partition layout
        pkg.ingest_structured_session(
            title="Session 05 · Checkout-Latency-Q4 Partition Pruning & FinOps Margin Sync",
            timestamp_label="7 days ago",
            age_days=7.0,
            transcript=(
                "[00:02] Maya Chen: Syncing with Priya Nair (FinOps Lead) and Devon Brooks (Staff SDE) on hidden-hour "
                "partition pruning in quick_lakehouse.checkout_telemetry_iceberg. Partitioning by day(event_ts) and "
                "checkout_region cut Athena Iceberg SQL scan volume by 74% per WBR diagnostic run. "
                "[05:40] Priya Nair (FinOps Lead): Confirmed infrastructure cost per 1,000 checkouts stays flat even "
                "with regional edge token pre-warming enabled for Checkout-Latency-Q4."
            ),
            facts=[],
        )

        # Session 6 (5 days ago) — Load test telemetry review (realistic operational session)
        pkg.ingest_structured_session(
            title="Session 06 · Cyber Week Canary Load Test & Edge Cache Telemetry",
            timestamp_label="5 days ago",
            age_days=5.0,
            transcript=(
                "[00:04] Devon Brooks (Staff SDE): In yesterday's 2x peak canary for Checkout-Latency-Q4, US-West-2 p99 "
                "dropped from 242ms to 194ms after enabling regional token session caching, leaving 14ms to shave to hit "
                "our p99 Checkout Latency <= 180ms SLA. EU-Central-1 telemetry joins in Athena Iceberg SQL are still "
                "waiting on Security Engineering to approve the LakeFormation Cross-Account Grants tag policy. "
                "[05:12] Maya Chen: Let's track the IAM tag policy ETA daily so we clear the EU partition blocker before Monday."
            ),
            facts=[],
        )

        # Session 7 (2 days ago) — Executive WBR format supersession + SDE collaboration
        pkg.ingest_structured_session(
            title="Session 07 · VP WBR Readout Format Change & SDE Escalation",
            timestamp_label="2 days ago",
            age_days=2.0,
            transcript=(
                "[00:01] Maya Chen: Format update from VP Commerce staff — stop generating 6-Page Narrative Memos for weekly "
                "reviews. Leadership now requires 3-Bullet BLUF + Weekly Delta Table for every status brief and escalation note. "
                "[03:40] Maya Chen: Collaborating closely with Devon Brooks (Staff SDE) on edge cache pre-warming and "
                "LakeFormation IAM unblocking, and committed to Tuesday 09:00 PT WBR Readout every week."
            ),
            facts=[
                {
                    "entity_type": "Preference",
                    "label": "3-Bullet BLUF + Weekly Delta Table",
                    "summary": "Format all status updates as a crisp 3-bullet BLUF followed by a compact WoW delta table.",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "doc_format",
                    "contradicts_label": "6-Page Narrative Memo",
                    "access_count": 10,
                    "pinned": True,
                    "evidence_quote": "Leadership now requires 3-Bullet BLUF + Weekly Delta Table for every status brief.",
                },
                {
                    "entity_type": "Person",
                    "label": "Devon Brooks (Staff SDE)",
                    "summary": "Partner Staff SDE leading checkout edge-cache warming and LakeFormation IAM unblocking.",
                    "relation": "COLLABORATES_WITH",
                    "access_count": 7,
                    "evidence_quote": "Collaborating closely with Devon Brooks (Staff SDE) on edge cache pre-warming.",
                },
                {
                    "entity_type": "Commitment",
                    "label": "Tuesday 09:00 PT WBR Readout",
                    "summary": "Weekly VP WBR executive readout deadline every Tuesday at 09:00 PT.",
                    "relation": "COMMITTED_TO",
                    "access_count": 8,
                    "evidence_quote": "Committed to Tuesday 09:00 PT WBR Readout.",
                },
            ],
        )
        return pkg

    def _seed_staff_sde(self) -> PersonaKnowledgeGraph:
        pkg = PersonaKnowledgeGraph(
            persona_id="staff_sde",
            user_name="Devon Brooks",
            role_title="Staff Software Development Engineer — Personal Memory & IR",
            org_unit="Amazon Quick Core Intelligence",
            root_summary="Staff SDE owning sub-2ms Personal Knowledge Graph retrieval and cross-region memory consistency.",
        )

        pkg.ingest_structured_session(
            title="Session 01 · Legacy OpenSearch Lexical Index Maintenance",
            timestamp_label="36 days ago",
            age_days=36.0,
            transcript=(
                "[00:01] Devon Brooks: Reviewing Q3 operational metrics for OpenSearch-Lexical-v2, our legacy BM25 keyword "
                "cluster backing workspace document lookup. All index plugins and custom tokenizers in OpenSearch-Lexical-v2 "
                "are built with Java 17 + Gradle, so prefer code samples in Java 17 + Gradle when proposing patches. "
                "[04:20] Maya Chen (Principal PM): Users are complaining that OpenSearch-Lexical-v2 forgets project pivots "
                "between chat sessions and surfaces outdated Q2/Q3 docs. Collaborating with Maya Chen (Principal PM) on "
                "scoping a next-generation persistent memory graph."
            ),
            facts=[
                {
                    "entity_type": "Project",
                    "label": "OpenSearch-Lexical-v2",
                    "summary": "Legacy keyword-only search service prior to Personal Knowledge Graph consolidation.",
                    "relation": "OWNS",
                    "slot_key": "primary_project",
                    "access_count": 2,
                    "evidence_quote": "Reviewing Q3 operational metrics for OpenSearch-Lexical-v2.",
                },
                {
                    "entity_type": "Preference",
                    "label": "Java 17 + Gradle",
                    "summary": "Historical JVM service implementation preference for legacy lexical search.",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "code_stack",
                    "access_count": 1,
                    "evidence_quote": "Prefer code samples in Java 17 + Gradle.",
                },
                {
                    "entity_type": "Person",
                    "label": "Maya Chen (Principal PM)",
                    "summary": "Principal PM partner defining cross-session personalization requirements.",
                    "relation": "COLLABORATES_WITH",
                    "access_count": 6,
                    "evidence_quote": "Collaborating with Maya Chen (Principal PM).",
                },
            ],
        )

        pkg.ingest_structured_session(
            title="Session 02 · Legacy JVM GC Profiling & Deprecation Plan",
            timestamp_label="26 days ago",
            age_days=26.0,
            transcript=(
                "[00:03] Devon Brooks: G1GC pause spikes on OpenSearch-Lexical-v2 hit 310ms under multi-tenant shard "
                "rebalancing in Java 17 + Gradle. Furthermore, flat keyword chunks have no directed edges to represent "
                "temporal supersession when a user changes roles or tooling preferences. We are freezing feature work "
                "on OpenSearch-Lexical-v2 and preparing a clean-sheet Personal Knowledge Graph runtime."
            ),
            facts=[],
        )

        pkg.ingest_structured_session(
            title="Session 03 · Legacy Gradle Plugin Postmortem & Synonym Drift",
            timestamp_label="18 days ago",
            age_days=18.0,
            transcript=(
                "[00:01] Devon Brooks: Postmortem on OpenSearch-Lexical-v2 synonym filter deployment: rolling restart "
                "of 24 data nodes via our Java 17 + Gradle shaded JAR took 47 minutes and still returned Q2/Q3 stale "
                "chunks ahead of Q4 charter updates because BM25 document frequency lacks SUPERSEDES edge semantics."
            ),
            facts=[],
        )

        pkg.ingest_structured_session(
            title="Session 04 · Architecture Cutover to QuickMemory-Hybrid-PKG",
            timestamp_label="9 days ago",
            age_days=9.0,
            transcript=(
                "[00:01] Devon Brooks: We completed service handoff — I transitioned ownership from OpenSearch-Lexical-v2 "
                "to QuickMemory-Hybrid-PKG. Do not generate Java 17 + Gradle boilerplate anymore: I switched preferred "
                "implementation stack from Java 17 + Gradle to Python 3.12 + Rust PyO3 for adjacency-list memory "
                "consolidation and zero-copy edge traversal. "
                "[05:18] Devon Brooks: We are tracking p99 Subgraph Retrieval <= 2.0ms across 2-hop neighborhoods, and "
                "multi-region active-active rollout is currently blocked by DynamoDB Global Table Replication Lag "
                "causing 420ms read-after-write skew in eu-central-1."
            ),
            facts=[
                {
                    "entity_type": "Project",
                    "label": "QuickMemory-Hybrid-PKG",
                    "summary": "2-hop Personal Knowledge Graph + BM25 contextual session engine with temporal supersession.",
                    "relation": "OWNS",
                    "slot_key": "primary_project",
                    "contradicts_label": "OpenSearch-Lexical-v2",
                    "access_count": 12,
                    "pinned": True,
                    "evidence_quote": "Transitioned ownership from OpenSearch-Lexical-v2 to QuickMemory-Hybrid-PKG.",
                },
                {
                    "entity_type": "Preference",
                    "label": "Python 3.12 + Rust PyO3",
                    "summary": "Write all new IR/graph engine code in zero-dependency Python 3.12 with Rust PyO3 hot loops.",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "code_stack",
                    "contradicts_label": "Java 17 + Gradle",
                    "access_count": 10,
                    "pinned": True,
                    "evidence_quote": "Switched preferred implementation stack from Java 17 + Gradle to Python 3.12 + Rust PyO3.",
                },
                {
                    "entity_type": "Metric",
                    "label": "p99 Subgraph Retrieval <= 2.0ms",
                    "summary": "Production latency budget for 2-hop personal knowledge graph neighborhood assembly.",
                    "relation": "TRACKS_METRIC",
                    "source_label": "QuickMemory-Hybrid-PKG",
                    "access_count": 9,
                    "evidence_quote": "Tracking p99 Subgraph Retrieval <= 2.0ms.",
                },
                {
                    "entity_type": "System",
                    "label": "DynamoDB Global Table Replication Lag",
                    "summary": "Cross-region replication skew (420ms spikes) causing stale read-after-write session state.",
                    "relation": "BLOCKED_BY",
                    "source_label": "QuickMemory-Hybrid-PKG",
                    "slot_key": "active_blocker",
                    "access_count": 7,
                    "evidence_quote": "Blocked by DynamoDB Global Table Replication Lag in eu-central-1.",
                },
            ],
        )

        pkg.ingest_structured_session(
            title="Session 05 · Read-Your-Writes Session Token & HLC Vector Clock Spike",
            timestamp_label="5 days ago",
            age_days=5.0,
            transcript=(
                "[00:02] Dr. Elena Rostova (Applied Scientist): Tested Ebbinghaus decay curves with frequency reinforcement "
                "on 50 synthetic multi-week transcripts. Adding Hybrid Logical Clocks (HLC) to session sticky tokens "
                "can fence stale replicas during DynamoDB Global Table Replication Lag windows without breaking our "
                "p99 Subgraph Retrieval <= 2.0ms budget in Python 3.12 + Rust PyO3."
            ),
            facts=[],
        )

        pkg.ingest_structured_session(
            title="Session 06 · PyO3 Adjacency Arena Benchmark & Memory Footprint",
            timestamp_label="3 days ago",
            age_days=3.0,
            transcript=(
                "[00:00] Devon Brooks: Benchmarked compact CSR adjacency arrays inside QuickMemory-Hybrid-PKG using "
                "Python 3.12 + Rust PyO3: 2-hop neighborhood expansion + tombstone bitmask filtering runs in sub-millisecond "
                "CPU time per user graph, leaving ample headroom under our p99 Subgraph Retrieval <= 2.0ms SLA."
            ),
            facts=[],
        )

        pkg.ingest_structured_session(
            title="Session 07 · Design Review Format & Production Readiness Commitment",
            timestamp_label="1 day ago",
            age_days=1.0,
            transcript=(
                "[00:01] Devon Brooks: Engineering communication preference — always structure technical design responses "
                "as Unified Diff + Big-O Complexity + Failure Modes rather than high-level prose. Collaborating with "
                "Dr. Elena Rostova (Applied Scientist) on Ebbinghaus decay calibration, and committed to Friday "
                "Production Readiness Review (PRR) for QuickMemory-Hybrid-PKG."
            ),
            facts=[
                {
                    "entity_type": "Preference",
                    "label": "Unified Diff + Big-O Complexity + Failure Modes",
                    "summary": "Every engineering answer must include concrete code/config, Big-O bounds, and failure modes.",
                    "relation": "PREFERS_FORMAT",
                    "slot_key": "doc_format",
                    "access_count": 9,
                    "pinned": True,
                    "evidence_quote": "Always structure technical design responses as Unified Diff + Big-O Complexity + Failure Modes.",
                },
                {
                    "entity_type": "Person",
                    "label": "Dr. Elena Rostova (Applied Scientist)",
                    "summary": "Applied Scientist co-designing Ebbinghaus retention curves and contradiction supersession.",
                    "relation": "COLLABORATES_WITH",
                    "access_count": 6,
                    "evidence_quote": "Collaborating with Dr. Elena Rostova (Applied Scientist) on Ebbinghaus decay calibration.",
                },
                {
                    "entity_type": "Commitment",
                    "label": "Friday Production Readiness Review (PRR)",
                    "summary": "Hard ORR / PRR sign-off gate this Friday for multi-region PKG rollout.",
                    "relation": "COMMITTED_TO",
                    "access_count": 7,
                    "evidence_quote": "Committed to Friday Production Readiness Review (PRR).",
                },
            ],
        )
        return pkg

    def get_persona(self, persona_id: str) -> PersonaKnowledgeGraph:
        pid = (persona_id or "principal_pm").strip()
        if pid not in self.personas:
            raise KeyError(f"Unknown persona '{pid}'. Valid personas: {', '.join(self.personas.keys())}")
        return self.personas[pid]

    def get_state(self, persona_id: str = "principal_pm") -> Dict[str, Any]:
        with self._lock:
            pkg = self.get_persona(persona_id)
            return {
                "active_persona": pkg.to_snapshot(),
                "personas": [
                    {
                        "persona_id": k,
                        "user_name": v.user_name,
                        "role_title": v.role_title,
                        "org_unit": v.org_unit,
                        "active_nodes": sum(1 for n in v.nodes.values() if n.status == "ACTIVE" and not n.forgotten),
                        "superseded_edges": sum(1 for e in v.edges.values() if e.status != "ACTIVE"),
                    }
                    for k, v in self.personas.items()
                ],
                "benchmarks": self.load_measured_benchmarks(),
            }

    def ingest_session(self, persona_id: str, session_title: str, transcript: str) -> Dict[str, Any]:
        with self._lock:
            pkg = self.get_persona(persona_id)
            ingest_result = pkg.ingest_live_transcript(session_title=session_title, transcript=transcript)
            return {
                "status": "consolidated",
                "persona_id": pkg.persona_id,
                "ingestion": ingest_result,
                "updated_persona": pkg.to_snapshot(),
            }

    def toggle_memory(self, persona_id: str, memory_id: str, action: str) -> Dict[str, Any]:
        with self._lock:
            pkg = self.get_persona(persona_id)
            res = pkg.toggle_memory_governance(memory_id=memory_id, action=action)
            return {
                "status": "updated",
                "persona_id": pkg.persona_id,
                **res,
                "updated_persona": pkg.to_snapshot(),
            }

    def ask_companion(self, persona_id: str, prompt: str) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            raise ValueError("Prompt is required.")

        with self._lock:
            pkg = self.get_persona(persona_id)
            retrieval = pkg.retrieve_2hop_context(clean_prompt)
            persona_name = pkg.user_name
            persona_role = pkg.role_title
            assembled_ctx = retrieval["assembled_context"]
            fallback_personalized = _build_deterministic_personalized_response(pkg, clean_prompt, retrieval)

        stateless_sys = (
            "You are a stateless, generic workplace AI assistant in a brand-new session with ZERO memory "
            "of who the user is, what projects they own, what SQL/code stack they use, or how they format updates. "
            "Keep your answer concise (under 140 words), generic, and explicitly note what missing context you would need."
        )
        personalized_sys = (
            f"You are Amazon Quick's QuickMemory Personalized Companion for {persona_name} ({persona_role}). "
            "Ground your response strictly in ACTIVE_PKG_NODES and ACTIVE_PKG_EDGES. "
            "Explicitly name the user's active owned Project, enforce every active Preference (exact SQL dialect, "
            "3-Bullet BLUF + Weekly Delta Table, or Python 3.12 + Rust PyO3 code + Big-O complexity + failure modes), "
            "surface relevant 2-hop Metric targets and System blockers, never use superseded facts, and cite [MEM-xx] and [EDGE-xx] inline. "
            "Keep under 220 words."
        )
        personalized_user = (
            f"{assembled_ctx}\n\n"
            f"FRESH_SESSION_USER_PROMPT:\n{clean_prompt}\n\n"
            "Respond directly to the user, strictly applying their active format/dialect preferences and citing [MEM-xx] and [EDGE-xx] provenance inline."
        )

        # Invoke live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) in parallel for Stateless vs QuickMemory Companion
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_stateless = pool.submit(_invoke_bedrock_nova_single, stateless_sys, clean_prompt, 220)
            fut_personal = pool.submit(_invoke_bedrock_nova_single, personalized_sys, personalized_user, 420)
            bedrock_stateless = fut_stateless.result()
            bedrock_personal = fut_personal.result()

        if bedrock_stateless:
            stateless_text = bedrock_stateless["text"]
            stateless_engine = bedrock_stateless["engine"]
            stateless_ms = bedrock_stateless["latency_ms"]
        else:
            stateless_text = _build_deterministic_stateless_response(clean_prompt)
            stateless_engine = "deterministic-stateless-baseline"
            stateless_ms = 1.2

        if bedrock_personal:
            personalized_text = bedrock_personal["text"]
            personalized_engine = bedrock_personal["engine"]
            personalized_ms = bedrock_personal["latency_ms"]
            # Ensure inline provenance badges are always present even if LLM abbreviated a tag
            valid_mem_ids = {n["memory_id"] for n in retrieval["retrieved_nodes"]}
            valid_edge_ids = {e["edge_id"] for e in retrieval["retrieved_edges"]}
            cited_mems = [m for m in MEM_CITE_RE.findall(personalized_text) if m in valid_mem_ids]
            cited_edges = [e for e in EDGE_CITE_RE.findall(personalized_text) if e in valid_edge_ids]
            if not cited_mems and retrieval["retrieved_nodes"]:
                anchor_mems = " ".join(f"[{n['memory_id']}]" for n in retrieval["retrieved_nodes"][:3])
                personalized_text += f"\n\nProvenance Nodes: {anchor_mems}"
            if not cited_edges and retrieval["retrieved_edges"]:
                anchor_edges = " ".join(f"[{e['edge_id']}]" for e in retrieval["retrieved_edges"][:2])
                personalized_text += f" · Edges: {anchor_edges}"
        else:
            personalized_text = fallback_personalized
            personalized_engine = "deterministic-pkg-synthesizer (offline fallback)"
            personalized_ms = round(retrieval["retrieval_latency_ms"] + 0.8, 2)

        valid_mem_ids = {n["memory_id"] for n in retrieval["retrieved_nodes"]}
        valid_edge_ids = {e["edge_id"] for e in retrieval["retrieved_edges"]}
        final_cited_mems = sorted(set(m for m in MEM_CITE_RE.findall(personalized_text) if m in valid_mem_ids))
        final_cited_edges = sorted(set(e for e in EDGE_CITE_RE.findall(personalized_text) if e in valid_edge_ids))

        return {
            "persona_id": persona_id,
            "user_name": persona_name,
            "prompt": clean_prompt,
            "stateless_response": {
                "text": stateless_text,
                "engine": stateless_engine,
                "latency_ms": stateless_ms,
                "preferences_applied": 0,
                "citations_count": 0,
            },
            "personalized_response": {
                "text": personalized_text,
                "engine": personalized_engine,
                "latency_ms": personalized_ms,
                "cited_memory_ids": final_cited_mems,
                "cited_edge_ids": final_cited_edges,
                "preferences_applied": sum(
                    1 for n in retrieval["retrieved_nodes"] if n["entity_type"] == "Preference"
                ),
            },
            "subgraph_trace": retrieval,
        }

    def health_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_nodes = sum(len(p.nodes) for p in self.personas.values())
            active_edges = sum(
                sum(1 for e in p.edges.values() if e.status == "ACTIVE")
                for p in self.personas.values()
            )
            superseded_edges = sum(
                sum(1 for e in p.edges.values() if e.status != "ACTIVE")
                for p in self.personas.values()
            )
            return {
                "personas_count": len(self.personas),
                "nodes_count": total_nodes,
                "active_edges_count": active_edges,
                "superseded_edges_count": superseded_edges,
            }
