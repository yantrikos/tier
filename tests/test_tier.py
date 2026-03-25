"""Tests for Tier Engine — Tier-based tool routing."""
import os, sys, json, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TierTestCase(unittest.TestCase):
    def setUp(self):
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.db_file.name
        self.db_file.close()
        from tier_engine.engine import TierEngine
        self.engine = TierEngine({"db_path": self.db_path})
    def tearDown(self):
        self.engine.close()
        for ext in ["", "-wal", "-shm"]:
            p = self.db_path + ext
            if os.path.exists(p): os.unlink(p)


class TestTierDetection(unittest.TestCase):
    def test_small_models(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("qwen3.5:0.8b"), ModelTier.SMALL)
        self.assertEqual(detect_tier("gemma3:2b"), ModelTier.SMALL)
        self.assertEqual(detect_tier("phi-3-mini-3b"), ModelTier.SMALL)

    def test_medium_models(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("qwen3.5:9b"), ModelTier.MEDIUM)
        self.assertEqual(detect_tier("llama-3.1-8b-instruct"), ModelTier.MEDIUM)
        self.assertEqual(detect_tier("haiku"), ModelTier.MEDIUM)

    def test_large_models(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("qwen3.5:27b"), ModelTier.LARGE)
        self.assertEqual(detect_tier("codellama-32b"), ModelTier.LARGE)
        self.assertEqual(detect_tier("sonnet"), ModelTier.LARGE)

    def test_xlarge_models(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("gpt-4"), ModelTier.XLARGE)
        self.assertEqual(detect_tier("claude"), ModelTier.XLARGE)
        self.assertEqual(detect_tier("llama-3.1-70b"), ModelTier.XLARGE)
        self.assertEqual(detect_tier("deepseek-chat"), ModelTier.XLARGE)

    def test_ollama_prefix_stripped(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("ollama/qwen3.5:9b"), ModelTier.MEDIUM)

    def test_unknown_defaults_medium(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("some-unknown-model"), ModelTier.MEDIUM)

    def test_tier_override(self):
        from tier_engine.detector import detect_tier
        from tier_engine.models import ModelTier
        self.assertEqual(detect_tier("tiny-model", {"tier": "XL"}), ModelTier.XLARGE)


class TestEmbeddings(unittest.TestCase):
    def test_tfidf_embedder(self):
        from tier_engine.embeddings import TFIDFEmbedder, cosine_similarity
        emb = TFIDFEmbedder()
        emb.fit(["read a file", "search the web", "write code"])
        v1 = emb.embed("read a file from disk")
        v2 = emb.embed("search the web for info")
        v3 = emb.embed("read a document")
        # v1 and v3 should be more similar than v1 and v2
        sim_13 = cosine_similarity(v1, v3)
        sim_12 = cosine_similarity(v1, v2)
        self.assertGreater(sim_13, sim_12)

    def test_cosine_similarity(self):
        from tier_engine.embeddings import cosine_similarity
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine_similarity([], []), 0.0)


class TestRouter(TierTestCase):
    def test_route_small_model(self):
        result = self.engine.route_and_format("read a file", "qwen3.5:0.8b")
        self.assertEqual(result["tier"], "S")
        self.assertEqual(result["format"], "mcq")
        self.assertIsNotNone(result["mcq_options"])
        self.assertIn("A", result["mcq_options"])

    def test_route_medium_model(self):
        result = self.engine.route_and_format("search for text in code", "llama-8b")
        self.assertEqual(result["tier"], "M")
        self.assertEqual(result["format"], "condensed")
        self.assertLessEqual(len(result["tools"]), 8)

    def test_route_large_model(self):
        result = self.engine.route_and_format("deploy an application", "qwen3.5:27b")
        self.assertEqual(result["tier"], "L")
        self.assertEqual(result["format"], "ranked")

    def test_route_xlarge_model(self):
        result = self.engine.route_and_format("anything", "gpt-4")
        self.assertEqual(result["tier"], "XL")
        self.assertEqual(result["format"], "full")

    def test_mcq_has_options(self):
        result = self.engine.route_and_format("read a file", "gemma3:2b")
        mcq = result["mcq_options"]
        self.assertGreater(len(mcq), 0)
        # Each option should have tool and description
        for label, opt in mcq.items():
            self.assertIn("tool", opt)
            self.assertIn("description", opt)

    def test_prompt_formatting(self):
        result = self.engine.route_and_format("search the web", "qwen3.5:0.8b")
        self.assertIn("Pick the best tool", result["prompt"])

    def test_relevance_ranking(self):
        result = self.engine.route_and_format("read a file from disk", "llama-8b")
        # file_read should be in top results
        self.assertIn("file_read", result["tools"][:3])

    def test_mcq_resolve(self):
        result = self.engine.route_and_format("read a file", "gemma3:2b")
        mcq = result["mcq_options"]
        tool = self.engine.resolve_mcq("A", result["tools"], mcq)
        self.assertIsNotNone(tool)


class TestCustomTools(TierTestCase):
    def test_register_custom_tool(self):
        self.engine.register_tool(
            name="deploy_function",
            description="Deploy a KubeFn function to the runtime",
            short_description="Deploy function",
            parameters={"name": "string", "code": "string"},
            category="kubefn",
        )
        tools = self.engine.router.list_tools()
        self.assertIn("deploy_function", tools)

    def test_custom_tool_persists(self):
        self.engine.register_tool("my_tool", "Does something cool", "Cool tool")
        # Reload engine
        self.engine.close()
        from tier_engine.engine import TierEngine
        self.engine = TierEngine({"db_path": self.db_path})
        tools = self.engine.router.list_tools()
        self.assertIn("my_tool", tools)


class TestUsageTracking(TierTestCase):
    def test_record_usage(self):
        self.engine.record_usage(
            intent="read a file",
            model_name="qwen:0.8b",
            tier="S",
            tool_selected="file_read",
            tool_executed="file_read",
            success=True,
            duration_ms=5,
        )
        stats = self.engine.stats()
        self.assertEqual(stats["total_routes"], 1)


class TestStats(TierTestCase):
    def test_stats(self):
        stats = self.engine.stats()
        self.assertGreater(stats["registered_tools"], 0)
        self.assertIn("routes_by_tier", stats)

    def test_health_check(self):
        health = self.engine.health_check()
        self.assertTrue(health["healthy"])
        self.assertEqual(health["engine"], "tier")

    def test_detect_tier_api(self):
        result = self.engine.detect_tier("qwen3.5:9b")
        self.assertEqual(result["tier"], "M")
        self.assertIn("max_tools", result)


if __name__ == "__main__":
    unittest.main()
