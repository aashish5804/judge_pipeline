"""
Judge and generator are configured independently on purpose: self-enhancement
bias (a judge favoring output from its own model family) is mitigated by
never letting the judge default to the generator's family.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    family: str          # e.g. "anthropic", "openai", "mock" — used for self-enhancement check
    temperature: float = 0.0


# Defaults. Override via CLI flags / env vars — never hardcode secrets here.
DEFAULT_JUDGE = ModelConfig(name="claude-opus-4-8", family="anthropic", temperature=0.0)
DEFAULT_GENERATOR_A = ModelConfig(name="gpt-4o", family="openai", temperature=0.7)
DEFAULT_GENERATOR_B = ModelConfig(name="gpt-4o-mini", family="openai", temperature=0.7)


def self_enhancement_risk(judge: ModelConfig, generator: ModelConfig) -> bool:
    """True if judge and generator share a model family — flags the risk instead
    of silently ignoring it."""
    return judge.family == generator.family
