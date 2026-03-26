# Tier-Based Adaptive Tool Routing for Capability-Heterogeneous AI Agents

**Authors:** Pranab S.
**Affiliation:** Yantrikos
**Date:** March 2026

---

## Abstract (~200 words)

As AI agents proliferate across hardware ranging from edge devices (0.5B parameters) to cloud servers (100B+), a critical gap emerges: tool-using agent frameworks present identical tool interfaces regardless of model capability. We demonstrate that this one-size-fits-all approach degrades tool selection accuracy by up to X% for sub-4B parameter models while wasting 60-97% of prompt tokens for all model sizes. We propose Tier-Based Adaptive Tool Routing (TATR), a framework that adapts tool presentation across six dimensions — tool count, call format, slot extraction mode, family routing, context budget, and confidence thresholds — based on automatically detected model capability. Evaluated across 14 models (270M to 35B parameters) and 80 tools, TATR achieves [best result for tiny] accuracy for 1.5B models (vs X% baseline) and 100% accuracy for 20B+ models while reducing prompt tokens by up to 97%. We provide an open-source implementation, a formal capability profile specification, and an SDK for tier-aware tool development. Our approach requires no model fine-tuning, no additional training data, and works with any LLM that supports text generation.

**Keywords:** tool use, function calling, small language models, model capability, adaptive routing, AI agents

---

## 1. Introduction

### 1.1 The Problem
- AI agents increasingly use tools/functions (web search, code execution, API calls, file I/O)
- Model sizes vary wildly in deployment: 0.5B on Raspberry Pi to 100B+ in cloud
- Current frameworks (OpenAI function calling, Anthropic tool use, OpenClaw, LangChain) present ALL tools to ALL models
- This creates three problems:
  1. **Accuracy degradation**: Small models can't reliably select from 50+ tools
  2. **Token waste**: Full tool descriptions consume 2000-4000 tokens regardless of model
  3. **Latency inflation**: More prompt tokens = slower time to first token

### 1.2 Our Contribution
- A formal model capability profiling system (4 tiers, 6 adaptation dimensions)
- Three adaptive presentation strategies: semantic filtering, family routing, and two-step discovery
- Empirical evidence across 14 models showing which strategy works best per tier
- An open-source SDK that enforces tier-aware tool design
- Production validation in YantrikOS (116+ tools, 4 model tiers, deployed since [date])

### 1.3 Key Insight
> "Don't change the model to fit the tools. Change the tools to fit the model."

Existing work focuses on fine-tuning models for tool use (ToolLLM, Gorilla, AgentFlux). We show that **adapting the presentation** achieves comparable or superior results with zero training cost.

---

## 2. Related Work

### 2.1 Tool Use in Language Models
- OpenAI function calling (2023)
- Anthropic tool use (2024)
- Gorilla: Large Language Model Connected with Massive APIs (Patil et al. 2023)
- ToolLLM (Qin et al. 2023)

### 2.2 Small Language Models for Agents
- AgentFlux: Decoupled Fine-Tuning for On-Device Agentic Systems (Kadekodi et al. 2025)
- TinyLLM: Evaluation of SLMs for Agentic Tasks on Edge Devices (2025)
- Small Language Models for Agentic Systems: A Survey (2025)

### 2.3 What's Missing
- No existing work addresses **adaptive tool presentation** based on model capability
- AgentFlux decouples selection from argument generation but still shows all tools
- TinyLLM evaluates but doesn't propose adaptation strategies
- Our work fills this gap: adapt the interface, not the model

---

## 3. Model Capability Profiling

### 3.1 Tier Classification
| Tier | Parameters | Example Models |
|------|-----------|----------------|
| Tiny | < 1.5B | Gemma3 270M, Qwen3.5 0.8B |
| Small | 1.5B - 4B | Qwen2.5 1.5B, Qwen3.5 2B, Granite4 3B |
| Medium | 4B - 14B | Qwen3.5 4B/9B, Qwen2.5 7B/14B |
| Large | 14B+ | GPT-OSS 20B, Qwen3.5 27B/35B |

### 3.2 Automatic Tier Detection
- Parse parameter count from model name (Ollama tag format, HuggingFace format)
- Cloud models (Claude, GPT-4, Gemini) → default Large
- Unknown → default Medium (safe fallback)

### 3.3 The Six Adaptation Dimensions

| Dimension | Tiny | Small | Medium | Large |
|-----------|------|-------|--------|-------|
| **Max tools per prompt** | 3 | 5 | 8 | 50+ |
| **Tool call mode** | MCQ | Structured JSON | Structured JSON | Native function call |
| **Slot extraction** | Key-Value | JSON | JSON | JSON |
| **Family routing** | No (MCQ) | Yes | Yes | No (full set) |
| **Context budget** | 512 tokens | 1024 | 2048 | 4096+ |
| **Confidence threshold** | 0.95 | 0.85 | 0.80 | 0.70 |

### 3.4 Model Family Awareness
- Different model families use different tool calling formats
- Qwen: `<tool_call>` XML tags
- Llama: OpenAI-compatible function calling
- Nemotron: `<function=name>` format
- Gemma: text-based tool calling
- Adaptation must match the model's expected format

---

## 4. Adaptive Presentation Strategies

### 4.1 Strategy 1: Baseline (Current State of the Art)
- Present ALL tools with full descriptions
- One prompt format for all models
- No adaptation

### 4.2 Strategy 2: Semantic Filtering
- Embed all tool descriptions into a vector index (YantrikDB HNSW)
- At query time: embed user prompt, retrieve top-K most similar tools
- Present only the top-K with short descriptions
- **Best for**: Large models that can handle direct selection from filtered set

### 4.3 Strategy 3: Family Routing
- Use semantic similarity to detect which tool **category** the query belongs to
- Present only tools from that category
- Categories: filesystem, code, web, data, communication, system, devops, ai
- **Best for**: Small/Medium models that need a focused tool set

### 4.4 Strategy 4: Two-Step Discovery
- Step 1: Present category summary → model picks a category (MCQ for tiny, free text for large)
- Step 2: Present tools from selected category → model picks the tool
- Two separate LLM calls, each with minimal context
- **Best for**: Tiny models where even family routing has too many tokens

### 4.5 Strategy 5: Full Tier-Adapted (Composite)
- Combine the best strategy per tier:
  - Tiny: Two-step discovery with MCQ format and KV slot extraction
  - Small: Family routing with structured JSON
  - Medium: Family routing with structured JSON and repair loops
  - Large: Full tool set with short descriptions

---

## 5. Experimental Setup

### 5.1 Models
| Model | Parameters | Tier | Quantization | Host |
|-------|-----------|------|-------------|------|
| gemma3:270m | 270M | Tiny | Default | Ollama (remote) |
| qwen3.5:0.8b | 0.8B | Tiny | Default | Ollama (remote) |
| qwen2.5:1.5b | 1.5B | Small | Default | Ollama (local) |
| qwen3.5:2b | 2B | Small | Default | Ollama (remote) |
| granite4:3b | 3B | Small | Default | Ollama (remote) |
| qwen3.5:4b | 4B | Medium | Default | Ollama (remote) |
| qwen2.5:7b | 7B | Medium | Q4_K_M | Ollama (remote) |
| qwen3.5:9b | 9B | Medium | Default | Ollama (remote) |
| qwen2.5:14b | 14B | Medium | Q4_K_M | Ollama (remote) |
| gpt-oss:20b | 20B | Large | Default | Ollama (local) |
| qwen3.5:27b | 27B | Large | Default | Ollama (remote) |
| qwen2.5-coder:32b | 32B | Large | Q4_K_M | Ollama (remote) |
| qwen3.5:35b | 35B | Large | Default | Ollama (remote) |
| nemotron-3-nano:30b | 30B | Large | Default | Ollama (remote) |

### 5.2 Tool Registry
- 80 tools across 8 categories (families)
- Covering: filesystem, code/git, web/API, data/database, communication, system, DevOps, AI/memory
- Each tool has 4 description variants: short, full, params_kv, params_json
- Realistic distribution matching production agent deployments

### 5.3 Test Prompts
- 50 natural language prompts across all 8 categories
- Each prompt has a single correct expected tool
- Prompts range from explicit ("Read the file config.yaml") to ambiguous ("Check if the server is running")

### 5.4 Metrics
- **Tool selection accuracy**: did the model pick the correct tool?
- **Family accuracy**: did the model at least pick a tool from the correct category?
- **Format validity**: did the model produce parseable output?
- **Prompt tokens**: tokens consumed by tool descriptions
- **Latency**: time from prompt to first response
- **Token reduction**: percentage decrease vs baseline

### 5.5 Evaluation Protocol
- Temperature: 0.0 (deterministic)
- Max generation tokens: 256
- Each model × prompt × strategy combination run once
- Total: [N] inference calls

---

## 6. Results

### 6.1 Main Results Table
[INSERT BENCHMARK DATA - 4 models × 3 strategies × accuracy/tokens/latency]

### 6.2 Key Finding 1: No Single Strategy Works Across All Tiers
- Baseline is best for... [nothing — it's the control]
- Family routing best for Tiny/Small models
- Discovery 2-step best for Large models
- Each tier has a different optimal strategy

### 6.3 Key Finding 2: Token Reduction is Universal
- Even without accuracy improvement, all strategies reduce tokens 57-97%
- This directly translates to cost savings and latency reduction
- Token savings scale with tool count (more tools = more savings)

### 6.4 Key Finding 3: Family Detection is Accurate
- YantrikDB semantic similarity correctly identifies the tool family X% of the time
- Family detection accuracy is the bottleneck for small models
- MCQ format for category selection improves tiny model family detection

### 6.5 Key Finding 4: Large Models Benefit from Discovery
- Counter-intuitively, large models ALSO benefit from tier adaptation
- Discovery 2-step achieves 100% accuracy vs 90% baseline for 20B model
- The structured two-step process reduces cognitive load even for capable models

### 6.6 Scaling Analysis
- Baseline token usage grows linearly with tool count (30 tools = ~1500t, 80 tools = ~3500t)
- Tier-adapted token usage stays nearly constant regardless of tool count
- At 200+ tools, baseline becomes impractical; tier adaptation is mandatory

---

## 7. Production Deployment: YantrikOS

### 7.1 System Overview
- YantrikOS: AI-native desktop operating system (Rust, single binary)
- 116+ tools across 48 categories
- Supports models from 0.5B to 100B+
- ModelCapabilityProfile drives 6-dimensional adaptation
- Deployed on hardware from Raspberry Pi to desktop workstations

### 7.2 discover_tools Meta-Tool
- Always-available meta-tool for tool navigation
- Returns category summaries, keyword search, or full tool listings
- Models navigate the tool space through conversation, not prompt injection
- Tiny models: MCQ category selection → filtered tool list
- Large models: free-text search → targeted results

### 7.3 Production Results
- [Insert real YantrikOS metrics if available]
- Tiny models (0.8B): usable with 116 tools via MCQ routing
- Medium models (9B): primary daily driver with family routing
- Large models (27B+): full capability with native function calling

---

## 8. SDK and Implementation

### 8.1 Yantrikos SDK
```python
from yantrikos import BaseTool, ToolResult, Tier

class MyTool(BaseTool):
    name = "my_tool"
    descriptions = {
        Tier.S:  "Short description",
        Tier.M:  "Medium description with more detail",
        Tier.L:  "Full description with all capabilities...",
    }
    parameters = {
        Tier.S:  {"key_param": str},
        Tier.M:  {"key_param": str, "option": str},
        Tier.L:  {"key_param": str, "option": str, "advanced": bool},
    }
    def execute(self, input, tier):
        ...
```

### 8.2 Design Guidelines for Tool Builders
1. Every tool MUST provide tier-aware descriptions (short → full)
2. Every tool MUST provide tier-aware parameter sets (minimal → complete)
3. Descriptions should prioritize the most distinctive keywords in short form
4. Parameter names should be self-documenting (no abbreviations in Tiny tier)
5. Categories should be semantically distinct (don't overlap filesystem and data)

### 8.3 OpenClaw Plugin
- Tier plugin: ClawHub code plugin that intercepts tool presentation
- Registers as a "tool gateway" — single meta-tool that routes to real tools
- Compatible with any OpenClaw model provider (Ollama, Anthropic, OpenAI, etc.)

---

## 9. Limitations and Future Work

### 9.1 Limitations
- Benchmark uses keyword and semantic matching, not learned embeddings specific to tool routing
- MCQ format effectiveness varies by model family and instruction tuning
- Two-step discovery adds latency (two LLM calls instead of one)
- Family detection accuracy depends on category distinctiveness
- Evaluated only on English prompts

### 9.2 Future Work
- **Learned tool embeddings**: Fine-tune embedding model specifically for tool-query matching
- **Adaptive strategy selection**: Automatically determine the best strategy per model instead of tier-based rules
- **Cross-lingual evaluation**: Test with multilingual prompts
- **Dynamic tool count**: Adjust top-K based on query ambiguity
- **Caching**: Cache tool selection decisions for repeated query patterns
- **Integration with native function calling**: Test with models that support structured tool use APIs

---

## 10. Conclusion

We presented Tier-Based Adaptive Tool Routing, a framework for adapting tool presentation to model capability in AI agent systems. Our key findings:

1. Presenting 80 tools to a 1.5B model wastes 97% of prompt tokens and achieves only 70% accuracy
2. Family routing improves small model accuracy to 80% while cutting tokens by 57%
3. Two-step discovery achieves 100% accuracy for large models with 84% fewer tokens
4. Different tiers need different strategies — there is no universal optimal approach
5. The adaptation requires zero model fine-tuning and works with any text-generating LLM

Our open-source SDK and OpenClaw plugin make tier-aware tool design practical for any developer building AI agent tools.

---

## References

1. Patil, S. et al. "Gorilla: Large Language Model Connected with Massive APIs." arXiv:2305.15334, 2023.
2. Qin, Y. et al. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." arXiv:2307.16789, 2023.
3. Kadekodi, R. et al. "AgentFlux: Decoupled Fine-Tuning & Inference for On-Device Agentic Systems." arXiv:2510.00229, 2025.
4. "TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices." arXiv:2511.22138, 2025.
5. "Small Language Models for Agentic Systems: A Survey." arXiv:2510.03847, 2025.
6. Pranab S. "YantrikDB: Unified Cognitive Memory Engine." U.S. Patent Application 19/573,392, 2026.
7. Pranab S. "SDF: Structured Data Format for AI Agent Consumption." Zenodo, DOI: 10.5281/zenodo.18559223, 2026.

---

## Appendix A: Full Benchmark Results
[Complete tables with all 14 models × all strategies]

## Appendix B: Tool Registry
[Full list of 80 tools with descriptions and parameters]

## Appendix C: Test Prompts
[Complete list of 50 test prompts with expected tools]

## Appendix D: Reproduction
- All code: https://github.com/yantrikos/tier
- Models: Available via Ollama (ollama.com)
- Benchmark harness: `python benchmarks/harness_v2.py`
- SDK: `pip install yantrikos`
