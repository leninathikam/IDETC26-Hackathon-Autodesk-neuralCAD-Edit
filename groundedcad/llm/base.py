"""LLM client abstractions."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import json_repair


@dataclass
class LLMResponse:
    text: str
    data: Optional[dict[str, Any]] = None
    token_counts: dict[str, float] = field(default_factory=dict)
    raw: Any = None


class LLMClient(ABC):
    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs
        self.total_tokens: dict[str, float] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "thinking_tokens": 0.0,
        }

    def _accumulate(self, counts: dict[str, float]) -> None:
        for k, v in counts.items():
            self.total_tokens[k] = self.total_tokens.get(k, 0.0) + float(v)

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        images: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        resp = self.complete(system=system, user=user + "\n\nReturn JSON only.", images=images)
        data = resp.data
        if data is None:
            data = parse_json_loose(resp.text)
        return data


def parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, str):
            return json.loads(repaired)
        raise


class MockLLMClient(LLMClient):
    """Deterministic offline client for tests and demos without API keys."""

    def __init__(self, model: str = "mock", scripted: Optional[dict[str, Any]] = None, **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self.scripted = scripted or {}
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, user: str, images: Optional[list[str]] = None) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "images": images})
        lower = (system + user).lower()
        if "ground" in lower or "temporal" in lower or "target" in lower:
            data = self.scripted.get(
                "grounder",
                {
                    "summary": "Apply the requested local edit",
                    "operation_hints": ["fillet"],
                    "dimensions": {},
                    "constraints": ["preserve overall shape"],
                    "ambiguities": [],
                    "confidence": 0.7,
                    "targets": [],
                },
            )
        elif "plan" in lower or "editspec" in lower or "tool_name" in lower:
            data = self.scripted.get(
                "planner",
                {
                    "operation": "custom",
                    "tool_name": "raw_cadquery",
                    "parameters": {},
                    "invariants": ["valid_brep"],
                    "success_checks": [
                        {"name": "valid_brep", "description": "valid", "check_type": "valid_brep"}
                    ],
                    "notes": "mock plan",
                    "script": None,
                },
            )
        elif "critic" in lower or "judge" in lower:
            data = self.scripted.get(
                "critic",
                {
                    "accept": True,
                    "instruction_score": 5,
                    "quality_score": 5,
                    "revision_advice": "",
                    "evidence": ["deterministic checks passed"],
                },
            )
        elif "my_cad_function" in lower or "cadquery 2" in lower:
            text = (
                "def my_cad_function(args):\n"
                "    import cadquery as cq\n"
                "    import os\n"
                "    shape = cq.importers.importStep(os.path.expanduser(args['input_file']))\n"
                "    return cq.Workplane('XY').newObject([shape]).translate((5, 0, 0))\n"
            )
            counts = {"input_tokens": 80, "output_tokens": 40, "thinking_tokens": 0}
            self._accumulate(counts)
            return LLMResponse(text=text, data=None, token_counts=counts)
        else:
            data = {"ok": True}
        text = json.dumps(data)
        counts = {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0}
        self._accumulate(counts)
        return LLMResponse(text=text, data=data, token_counts=counts)


class OpenAIClient(LLMClient):
    def complete(self, *, system: str, user: str, images: Optional[list[str]] = None) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI()
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for img in images or []:
            p = Path(img)
            if p.exists():
                import base64

                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **{k: v for k, v in self.kwargs.items() if k in {"temperature", "max_tokens"}},
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        counts = {
            "input_tokens": float(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": float(getattr(usage, "completion_tokens", 0) or 0),
            "thinking_tokens": 0.0,
        }
        self._accumulate(counts)
        return LLMResponse(text=text, data=None, token_counts=counts, raw=resp)


class AnthropicClient(LLMClient):
    def complete(self, *, system: str, user: str, images: Optional[list[str]] = None) -> LLMResponse:
        import base64

        from anthropic import Anthropic

        client = Anthropic()
        content: list[dict[str, Any]] = []
        for img in images or []:
            p = Path(img)
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )
        content.append({"type": "text", "text": user})
        resp = client.messages.create(
            model=self.model,
            max_tokens=int(self.kwargs.get("max_tokens", 4096)),
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        counts = {
            "input_tokens": float(getattr(resp.usage, "input_tokens", 0) or 0),
            "output_tokens": float(getattr(resp.usage, "output_tokens", 0) or 0),
            "thinking_tokens": 0.0,
        }
        self._accumulate(counts)
        return LLMResponse(text=text, data=None, token_counts=counts, raw=resp)


class GeminiClient(LLMClient):
    def complete(self, *, system: str, user: str, images: Optional[list[str]] = None) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client()
        parts: list[Any] = [types.Part.from_text(text=f"{system}\n\n{user}")]
        for img in images or []:
            p = Path(img)
            if p.exists():
                mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
        resp = client.models.generate_content(model=self.model, contents=parts)
        text = resp.text or ""
        counts = {"input_tokens": 0.0, "output_tokens": 0.0, "thinking_tokens": 0.0}
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            counts["input_tokens"] = float(getattr(usage, "prompt_token_count", 0) or 0)
            counts["output_tokens"] = float(getattr(usage, "candidates_token_count", 0) or 0)
        self._accumulate(counts)
        return LLMResponse(text=text, data=None, token_counts=counts, raw=resp)


def build_llm_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    provider = provider.lower()
    if provider in {"mock", "offline"}:
        return MockLLMClient(model=model, **kwargs)
    if provider in {"openai", "gpt"}:
        return OpenAIClient(model=model, **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicClient(model=model, **kwargs)
    if provider in {"gemini", "google"}:
        return GeminiClient(model=model, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")


def auto_client_from_env(default_provider: str = "mock") -> LLMClient:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    max_tokens = int(os.getenv("GROUNDEDCAD_MAX_TOKENS", "8192"))
    if os.getenv("GROUNDEDCAD_PROVIDER"):
        provider = os.environ["GROUNDEDCAD_PROVIDER"]
        model = os.getenv("GROUNDEDCAD_MODEL", "mock")
        return build_llm_client(provider, model, max_tokens=max_tokens)
    if os.getenv("OPENAI_API_KEY"):
        return build_llm_client("openai", os.getenv("GROUNDEDCAD_MODEL", "gpt-4.1"), max_tokens=max_tokens)
    if os.getenv("ANTHROPIC_API_KEY"):
        return build_llm_client(
            "anthropic",
            os.getenv("GROUNDEDCAD_MODEL", "claude-sonnet-5"),
            max_tokens=max_tokens,
        )
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return build_llm_client("gemini", os.getenv("GROUNDEDCAD_MODEL", "gemini-2.5-pro"))
    return build_llm_client(default_provider, "mock")
