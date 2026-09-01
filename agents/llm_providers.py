"""
LLM-agnostic interface, per your review point: the reasoning/report layer
should not be hardcoded to one vendor. Any provider below can be swapped in
without touching decision_agent.py -- it only calls .generate(prompt).

Includes a RuleBasedProvider that needs NO API key and NO internet access,
so decision_agent.py works out of the box for grading/demo purposes even
without any LLM subscription. Swap to a real provider by changing one line.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return a text completion for the given prompt."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Requires: pip install anthropic --break-system-packages
    and an ANTHROPIC_API_KEY environment variable."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """Requires: pip install openai --break-system-packages
    and an OPENAI_API_KEY environment variable."""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """For local open models via Ollama running on your machine.
    Requires: Ollama installed and running (https://ollama.com), and the
    model already created/pulled, e.g. `ollama create phi4-telecom-rootcause -f Modelfile`.

    NOTE: `num_predict` and `temperature` are passed explicitly in the
    `options` dict below -- without this, Ollama uses its own default
    max-token limit, which can silently truncate structured outputs (e.g.
    stopping right after "CATEGORY: capacity_creep" before CONFIDENCE/
    REASONING are ever generated). 300 tokens is comfortable headroom for
    a CATEGORY + CONFIDENCE + REASONING (2-3 sentence) response.
    """

    def __init__(self, model: str = "phi4-telecom-rootcause", host: str = "http://localhost:11434",
                 num_predict: int = 300, temperature: float = 0.3):
        self.model = model
        self.host = host
        self.num_predict = num_predict
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.num_predict,
                    "temperature": self.temperature,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class Phi4LocalProvider(LLMProvider):
    """
    Runs your fine-tuned Phi-4 Mini locally via llama-cpp-python, using the
    quantized GGUF file produced by finetune/finetune_phi4_colab.ipynb.
    CPU-only, no internet, no API key -- this is what makes the fine-tuned
    model usable on an 8GB RAM machine with no GPU.

    Requires: pip install llama-cpp-python --break-system-packages
    Expects the GGUF file at agents/models/phi4-telecom-rootcause-Q4_K_M.gguf
    by default (move the file downloaded from Colab there, or pass a custom
    model_path).
    """

    def __init__(self, model_path: str = "models/phi4-telecom-rootcause-Q4_K_M.gguf",
                 n_ctx: int = 2048, n_threads: int = None):
        from llama_cpp import Llama
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model not found at '{model_path}'. Run finetune/finetune_phi4_colab.ipynb "
                f"first, download the resulting .gguf file, and place it at this path (or pass "
                f"model_path explicitly)."
            )
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def generate(self, prompt: str) -> str:
        # Phi models use a <|user|>/<|assistant|> chat format -- this must
        # match whatever format the fine-tuning notebook trained on (it uses
        # tokenizer.apply_chat_template, so this is the standard Phi format).
        formatted = f"<|system|>\nYou are a telecom network root cause diagnosis assistant.<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        output = self.llm(formatted, max_tokens=400, stop=["<|end|>"], temperature=0.3)
        return output["choices"][0]["text"].strip()


class RuleBasedProvider(LLMProvider):
    """
    No API, no internet, no cost. Produces a templated explanation from the
    same evidence dict an LLM would receive. Weaker prose than a real LLM,
    but deterministic and always available -- good default so the whole
    pipeline runs end-to-end even with zero API keys configured, and useful
    as a baseline to compare LLM-generated explanations against.
    """

    def generate(self, prompt: str) -> str:
        # The prompt built by decision_agent.py embeds the evidence as
        # labeled lines; parse them back out for a simple templated summary.
        lines = [l for l in prompt.split("\n") if l.strip().startswith("-")]
        if not lines:
            return "Insufficient structured evidence to generate an explanation."
        top = lines[0].replace("-", "").strip()
        return (
            f"Rule-based summary (no LLM configured): the strongest signal was '{top}'. "
            f"Full evidence considered: {'; '.join(l.replace('-', '').strip() for l in lines)}."
        )


def get_provider(name: str = "rule_based", **kwargs) -> LLMProvider:
    """Factory -- this is the one line you change to swap providers."""
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "phi4_local": Phi4LocalProvider,
        "rule_based": RuleBasedProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(providers)}")
    return providers[name](**kwargs)


if __name__ == "__main__":
    provider = get_provider("rule_based")
    sample_prompt = (
        "Evidence:\n"
        "- Battery level dropped from 100% to 40% over the last 3 hours\n"
        "- Grid voltage sagging (215V, below normal 230V)\n"
        "- Signal strength stable\n"
        "- Traffic normal\n"
        "What is the likely cause?"
    )
    print(provider.generate(sample_prompt))
