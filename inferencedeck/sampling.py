from __future__ import annotations

from typing import Any

# ponytail: curated community-convention presets, not learned per-model optima.
# Each intent maps to sampling params plus a one-line rationale per field so the
# UI can explain why. Presets are starting points the user then tunes.
SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "coding": {
        "label": "Coding",
        "description": "Tight, repeatable output for code and structured formats.",
        "params": {"temperature": 0.2, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                   "repeat_penalty": 1.1, "repeat_last_n": 64},
        "rationale": {
            "temperature": "Low temperature keeps syntax and logic stable.",
            "top_k": "Moderate top-k trims unlikely tokens without starving valid ones.",
            "top_p": "High top-p still allows the few good completions code needs.",
            "min_p": "Small min-p floor drops long-tail noise.",
            "repeat_penalty": "Mild penalty curbs loops without breaking boilerplate.",
        },
    },
    "factual": {
        "label": "Factual",
        "description": "Grounded, low-variance answers for retrieval and reasoning.",
        "params": {"temperature": 0.3, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
                   "repeat_penalty": 1.1, "repeat_last_n": 64},
        "rationale": {
            "temperature": "Low temperature favors the most supported answer.",
            "top_k": "Keeps the candidate set focused on likely facts.",
            "top_p": "Slightly tighter top-p reduces speculative wording.",
            "min_p": "Floors out improbable tokens.",
            "repeat_penalty": "Discourages restating the same phrasing.",
        },
    },
    "balanced": {
        "label": "Balanced",
        "description": "General assistant default — coherent but not rigid.",
        "params": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.05,
                   "repeat_penalty": 1.1, "repeat_last_n": 64},
        "rationale": {
            "temperature": "Mid temperature balances reliability and variety.",
            "top_k": "Standard top-k for conversational range.",
            "top_p": "Nucleus sampling keeps replies natural.",
            "min_p": "Trims the noisy tail.",
            "repeat_penalty": "Keeps multi-turn replies from looping.",
        },
    },
    "creative": {
        "label": "Creative",
        "description": "Varied, surprising output for prose and brainstorming.",
        "params": {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "min_p": 0.02,
                   "repeat_penalty": 1.05, "repeat_last_n": 256},
        "rationale": {
            "temperature": "High temperature widens word choice for novelty.",
            "top_k": "Large top-k admits more unusual tokens.",
            "top_p": "High top-p keeps the long tail in play.",
            "min_p": "Low min-p preserves rare but interesting tokens.",
            "repeat_penalty": "Light penalty over a longer window avoids stale phrasing.",
        },
    },
}


def list_sampling_intents() -> list[dict[str, str]]:
    return [{"key": key, "label": preset["label"], "description": preset["description"]}
            for key, preset in SAMPLING_PRESETS.items()]


def suggest_sampling(intent: str) -> dict[str, Any]:
    preset = SAMPLING_PRESETS.get((intent or "").lower())
    if not preset:
        return {"success": False, "error": f"Unknown sampling intent: {intent!r}",
                "intents": [key for key in SAMPLING_PRESETS]}
    return {"success": True, "intent": intent, "label": preset["label"],
            "description": preset["description"], "params": dict(preset["params"]),
            "rationale": dict(preset["rationale"])}


def demo() -> None:
    assert suggest_sampling("coding")["params"]["temperature"] == 0.2
    assert suggest_sampling("creative")["params"]["temperature"] == 1.0
    assert suggest_sampling("nope")["success"] is False
    assert len(list_sampling_intents()) == len(SAMPLING_PRESETS)
    print("sampling demo OK")


if __name__ == "__main__":
    demo()
