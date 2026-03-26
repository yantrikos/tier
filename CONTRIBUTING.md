# Contributing to Tier

Thank you for your interest in making AI agents work better across model sizes.

## Ways to Contribute

### 1. Build Tier-Aware Tools

The most impactful contribution is building tools that follow the tier specification.

```bash
pip install yantrikos-sdk
```

```python
from yantrikos import BaseTool, ToolResult, Tier, register

@register
class YourTool(BaseTool):
    name = "your_tool"
    category = "your_category"  # filesystem, web, code, data, etc.

    descriptions = {
        Tier.S:  "2-3 words that uniquely identify this tool",
        Tier.M:  "One sentence, core functionality",
        Tier.L:  "Full description with key capabilities",
        Tier.XL: "Complete description with all options and edge cases",
    }

    parameters = {
        Tier.S:  {"essential_param": str},
        Tier.M:  {"essential_param": str, "common_option": str},
        Tier.L:  {"essential_param": str, "common_option": str, "advanced": bool},
        Tier.XL: {"essential_param": str, "common_option": str, "advanced": bool,
                  "timeout": int, "format": str},
    }

    def execute(self, input: dict, tier: Tier) -> ToolResult:
        # Tier-aware execution
        ...
```

#### Tool Design Checklist

- [ ] Tier.S description is under 50 characters
- [ ] Tier.S has at most 2 parameters
- [ ] First parameter is the most important one
- [ ] Descriptions are discriminative (distinguish from similar tools)
- [ ] Category is one of: filesystem, code, web, data, communication, system, devops, ai
- [ ] Tool works correctly when called with only Tier.S parameters
- [ ] Tested with a 1.5B model via Ollama native tool calling

### 2. Improve Benchmark Data

Run benchmarks on models we haven't tested:

```bash
git clone https://github.com/yantrikos/tier
cd tier
pip install yantrikos-sdk yantrikdb sentence-transformers

# Run with your model
python benchmarks/harness_v3.py --models "your-model:size"
```

Share results by opening an issue with:
- Model name and size
- Ollama version
- Hardware specs
- The generated JSONL file

Models we'd especially like data for:
- Llama 3.x (1B, 3B, 8B, 70B)
- Gemma 3 (2B, 4B, 12B, 27B)
- Phi 4 (3.8B, 14B)
- Mistral/Mixtral
- Any non-English models

### 3. Improve Routing

The family detection bottleneck (54-64% accuracy) is the biggest open challenge. Ideas to explore:

- Better embedding models for tool ranking
- LLM-based family routing with self-correction
- Hybrid strategies combining semantic + keyword + usage patterns
- Multi-turn discovery evaluation

### 4. Add Tests

```bash
python -m unittest discover tests -v
```

We welcome tests for:
- Edge cases in tier detection (unusual model names)
- Tools with overlapping categories
- Parameter validation
- Native tool format correctness

### 5. Documentation

- Improve code examples in README
- Add tutorials for specific frameworks (LangChain, CrewAI, LlamaIndex)
- Translate documentation

## Development Setup

```bash
git clone https://github.com/yantrikos/tier
cd tier
pip install -e .
pip install yantrikos-sdk

# Run tests
python -m unittest discover tests -v

# Run benchmarks (requires Ollama)
python benchmarks/harness_v3.py --quick
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`python -m unittest discover tests -v`)
5. Commit with a clear message
6. Open a pull request

## Code Style

- Python 3.9+ compatible (no `X | Y` union types)
- No external dependencies in core SDK (stdlib only)
- Type hints where they aid readability
- Docstrings for public methods

## Reporting Issues

When reporting bugs, include:
- Python version
- yantrikos-sdk version
- Ollama version (if benchmark-related)
- Model name and size
- Minimal reproduction steps

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
