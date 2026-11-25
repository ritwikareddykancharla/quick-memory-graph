"""Comprehensive Unit Tests for QuickMemory Engine & CloudFront `/api/*` Lambda Router."""
from __future__ import annotations

import json
import os
import unittest

from quick_memory.graph_engine import QuickMemoryEngine, compute_ebbinghaus_retention
from quick_memory.server import ENGINE, lambda_handler, route_api_request


class QuickMemoryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["QUICKMEMORY_OFFLINE_LLM"] = "1"
        self.engine = QuickMemoryEngine()
        ENGINE.reset()

    def test_01_ebbinghaus_temporal_decay_and_frequency_boost(self) -> None:
        fresh = compute_ebbinghaus_retention(age_days=1.0, access_count=1)
        aged_low_freq = compute_ebbinghaus_retention(age_days=30.0, access_count=1)
        aged_high_freq = compute_ebbinghaus_retention(age_days=30.0, access_count=15)
        pinned = compute_ebbinghaus_retention(age_days=60.0, access_count=0, pinned=True)
        forgotten = compute_ebbinghaus_retention(age_days=1.0, access_count=20, pinned=True, forgotten=True)

        self.assertGreater(fresh, aged_low_freq)
        self.assertGreater(aged_high_freq, aged_low_freq)
        self.assertEqual(pinned, 1.0)
        self.assertEqual(forgotten, 0.0)

    def test_02_seed_personas_resolve_temporal_contradictions(self) -> None:
        snap = self.engine.get_state("principal_pm")["active_persona"]
        nodes_by_label = {n["label"]: n for n in snap["nodes"]}

        self.assertEqual(nodes_by_label["Payments-v1"]["status"], "SUPERSEDED")
        self.assertEqual(nodes_by_label["Checkout-Latency-Q4"]["status"], "ACTIVE")
        self.assertEqual(nodes_by_label["Redshift Warehouse SQL"]["status"], "SUPERSEDED")
        self.assertEqual(nodes_by_label["Athena Iceberg SQL"]["status"], "ACTIVE")
        self.assertEqual(nodes_by_label["6-Page Narrative Memo"]["status"], "SUPERSEDED")
        self.assertEqual(nodes_by_label["3-Bullet BLUF + Weekly Delta Table"]["status"], "ACTIVE")

        supersedes_edges = [e for e in snap["active_edges"] if e["relation"] == "SUPERSEDES"]
        self.assertGreaterEqual(len(supersedes_edges), 3)

    def test_03_live_transcript_ingestion_supersedes_prior_project_and_dialect(self) -> None:
        res = self.engine.ingest_session(
            persona_id="principal_pm",
            session_title="Session 04 · Cyber Week Cutover",
            transcript=(
                "Switched project from Checkout-Latency-Q4 to Global-OneClick-2026. "
                "Migrated preferred SQL dialect from Athena Iceberg SQL to DuckDB Parquet Local SQL. "
                "Collaborating with Raj Malhotra (Principal TPM). "
                "Blocked by CloudFront Regional Edge Cache Warmup."
            ),
        )
        self.assertEqual(res["status"], "consolidated")
        self.assertGreaterEqual(len(res["ingestion"]["superseded_edges"]), 2)

        updated = {n["label"]: n for n in res["updated_persona"]["nodes"]}
        self.assertEqual(updated["Checkout-Latency-Q4"]["status"], "SUPERSEDED")
        self.assertEqual(updated["Global-OneClick-2026"]["status"], "ACTIVE")
        self.assertEqual(updated["Athena Iceberg SQL"]["status"], "SUPERSEDED")
        self.assertEqual(updated["DuckDB Parquet Local SQL"]["status"], "ACTIVE")
        self.assertIn("Raj Malhotra (Principal TPM)", updated)
        self.assertIn("CloudFront Regional Edge Cache Warmup", updated)

    def test_04_two_hop_retrieval_excludes_superseded_contradictions(self) -> None:
        pkg = self.engine.get_persona("principal_pm")
        trace = pkg.retrieve_2hop_context(
            "Draft my Tuesday WBR update and write the p99 checkout latency SQL query."
        )
        retrieved_labels = {n["label"] for n in trace["retrieved_nodes"]}
        self.assertIn("Checkout-Latency-Q4", retrieved_labels)
        self.assertIn("Athena Iceberg SQL", retrieved_labels)
        self.assertIn("3-Bullet BLUF + Weekly Delta Table", retrieved_labels)
        # 2-hop neighbors attached to Checkout-Latency-Q4 must be traversed
        self.assertIn("p99 Checkout Latency <= 180ms", retrieved_labels)
        self.assertIn("LakeFormation Cross-Account Grants", retrieved_labels)

        # Superseded entities must NEVER leak into assembled active context
        self.assertNotIn("Payments-v1", retrieved_labels)
        self.assertNotIn("Redshift Warehouse SQL", retrieved_labels)
        self.assertNotIn("6-Page Narrative Memo", retrieved_labels)
        self.assertGreater(trace["token_reduction_pct"], 50.0)

    def test_05_memory_governance_pin_forget_and_restore(self) -> None:
        snap = self.engine.get_state("principal_pm")["active_persona"]
        priya = next(n for n in snap["nodes"] if "Priya Nair" in n["label"])
        mid = priya["memory_id"]

        pin_res = self.engine.toggle_memory("principal_pm", mid, "pin")
        self.assertTrue(pin_res["memory"]["pinned"])
        self.assertEqual(pin_res["memory"]["retention_score"], 1.0)

        forget_res = self.engine.toggle_memory("principal_pm", mid, "forget")
        self.assertTrue(forget_res["memory"]["forgotten"])
        self.assertEqual(forget_res["memory"]["status"], "TOMBSTONED")
        self.assertGreaterEqual(len(forget_res["affected_edge_ids"]), 1)

        # Forgotten entity must never appear in 2-hop retrieval
        trace = self.engine.get_persona("principal_pm").retrieve_2hop_context("margin attribution with Priya Nair")
        retrieved_ids = {n["memory_id"] for n in trace["retrieved_nodes"]}
        self.assertNotIn(mid, retrieved_ids)

        restore_res = self.engine.toggle_memory("principal_pm", mid, "restore")
        self.assertFalse(restore_res["memory"]["forgotten"])
        self.assertEqual(restore_res["memory"]["status"], "ACTIVE")

    def test_06_companion_ask_returns_side_by_side_with_provenance_citations(self) -> None:
        out = self.engine.ask_companion(
            persona_id="principal_pm",
            prompt="Prepare my weekly WBR status brief and p99 checkout latency SQL query.",
        )
        self.assertIn("stateless_response", out)
        self.assertIn("personalized_response", out)
        self.assertGreaterEqual(len(out["personalized_response"]["cited_memory_ids"]), 2)
        self.assertIn("Athena Iceberg SQL", out["personalized_response"]["text"])

    def test_07_api_health_and_state_routes(self) -> None:
        code_h, health = route_api_request("GET", "/api/health", {}, {})
        self.assertEqual(code_h, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "quick-memory-graph")
        self.assertEqual(health["bedrock_model"], "us.amazon.nova-lite-v1:0")
        self.assertGreater(health["nodes_count"], 15)

        code_s, state = route_api_request("GET", "/api/state", {"persona": "staff_sde"}, {})
        self.assertEqual(code_s, 200)
        self.assertEqual(state["active_persona"]["user_name"], "Devon Brooks")

    def test_08_api_ingest_ask_toggle_forget_and_reset_routes(self) -> None:
        code_i, ing = route_api_request(
            "POST",
            "/api/sessions/ingest",
            {},
            {
                "persona": "staff_sde",
                "session_title": "Session 04 · Rust SIMD Hotpath",
                "transcript": "Committed to shipping AVX-512 bitset filter before Tuesday load test.",
            },
        )
        self.assertEqual(code_i, 200)
        new_mem_id = ing["ingestion"]["extracted_nodes"][0]["memory_id"]

        code_f, fgt = route_api_request(
            "DELETE",
            "/api/memory/forget",
            {"persona": "staff_sde", "memory_id": new_mem_id},
            {},
        )
        self.assertEqual(code_f, 200)
        self.assertTrue(fgt["memory"]["forgotten"])

        code_a, ask = route_api_request(
            "POST",
            "/api/companion/ask",
            {},
            {"persona": "staff_sde", "prompt": "Summarize our active PKG architecture and replication blocker."},
        )
        self.assertEqual(code_a, 200)
        self.assertIn("QuickMemory-Hybrid-PKG", ask["personalized_response"]["text"])

        code_r, rst = route_api_request("POST", "/api/reset", {}, {"persona": "principal_pm"})
        self.assertEqual(code_r, 200)
        self.assertEqual(rst["status"], "reset_complete")

    def test_09_lambda_handler_cloudfront_event_envelope(self) -> None:
        event = {
            "rawPath": "/api/companion/ask",
            "requestContext": {"http": {"method": "POST"}},
            "queryStringParameters": None,
            "body": json.dumps(
                {
                    "persona": "principal_pm",
                    "prompt": "Write the p99 checkout latency SQL query for Tuesday WBR.",
                }
            ),
            "isBase64Encoded": False,
        }
        resp = lambda_handler(event)
        self.assertEqual(resp["statusCode"], 200)
        body = json.loads(resp["body"])
        self.assertEqual(body["persona_id"], "principal_pm")
        self.assertTrue(len(body["personalized_response"]["cited_memory_ids"]) >= 1)

    def test_10_validation_errors_handled_cleanly(self) -> None:
        c1, _ = route_api_request("POST", "/api/companion/ask", {}, {"persona": "principal_pm", "prompt": "   "})
        self.assertEqual(c1, 400)
        c2, _ = route_api_request("GET", "/api/state", {"persona": "nonexistent_role"}, {})
        self.assertEqual(c2, 404)
        c3, _ = route_api_request("POST", "/api/memory/toggle", {}, {"persona": "principal_pm", "memory_id": "USER-00", "action": "forget"})
        self.assertEqual(c3, 400)


if __name__ == "__main__":
    unittest.main()
