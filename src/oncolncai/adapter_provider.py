"""Local PEFT adapter exposed through the existing structured-provider interface."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from oncolncai.providers import LLMProviderError


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("adapter response must be a JSON object")
    return value


class FineTunedAdapterProvider:
    """Generate schema-validated JSON with a local LoRA/QLoRA adapter.

    Heavy ML dependencies are imported only when the default generator is first
    used. Tests can inject a lightweight generator, and application extraction
    remains coupled only to ``LLMProvider.generate_structured``.
    """

    provider_name = "transformers_peft"

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | Path,
        max_new_tokens: int = 512,
        generator: Callable[[str], str] | None = None,
    ) -> None:
        if not base_model.strip():
            raise ValueError("base_model cannot be blank")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self._generator = generator

    def _load_generator(self) -> Callable[[str], str]:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not hasattr(torch.nn.Module, "set_submodule"):
                def set_submodule(module: torch.nn.Module, target: str, replacement: torch.nn.Module) -> None:
                    parent_name, _, child_name = target.rpartition(".")
                    parent = module.get_submodule(parent_name) if parent_name else module
                    setattr(parent, child_name, replacement)
                torch.nn.Module.set_submodule = set_submodule

            tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=quantization,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(base, self.adapter_path)
            model.eval()

            def generate(prompt: str) -> str:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

            return generate
        except Exception as exc:
            raise LLMProviderError(self.provider_name, "load_error", str(exc) or type(exc).__name__, False) from exc

    def generate_structured(self, *, prompt: str, output_schema: type[BaseModel]) -> Any:
        if self._generator is None:
            self._generator = self._load_generator()
        try:
            return output_schema.model_validate(_json_object(self._generator(prompt))).model_dump(mode="json")
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(self.provider_name, "invalid_response", str(exc) or type(exc).__name__, False) from exc

