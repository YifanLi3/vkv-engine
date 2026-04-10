"""
SamplingParams — generation parameters.

Directly matches nano-vLLM's sampling_params.py.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Parameters for text generation sampling.
    Same fields as nano-vLLM's SamplingParams.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
