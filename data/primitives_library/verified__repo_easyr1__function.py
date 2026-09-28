import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]
BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        responses = data.batch["responses"]
        response_lengths = torch.sum(data.batch["response_mask"], dim=-1)
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        metrics = defaultdict(list)

        for index in range(len(data)):
            length = int(response_lengths[index].item())
            tokens = responses[index][:length]
            response = self.tokenizer.decode(
                tokens,
                skip_special_tokens=self.config.skip_special_tokens,
            )
            score = self.reward_fn(
                {
                    "response": response,
                    "response_length": length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][index],
                }
            )

            rewards[index, length - 1] = score["overall"]
            for key, value in score.items():
                metrics[key].append(value)

        return rewards, metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        responses = data.batch["responses"]
        response_lengths = torch.sum(data.batch["response_mask"], dim=-1)
        inputs = []

        for index in range(len(data)):
            length = int(response_lengths[index].item())
            tokens = responses[index][:length]
            response = self.tokenizer.decode(
                tokens,
                skip_special_tokens=self.config.skip_special_tokens,
            )
            inputs.append(
                {
                    "response": response,
                    "response_length": length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][index],
                }
            )

        scores = self.reward_fn(inputs)
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        metrics = defaultdict(list)

        for index, score in enumerate(scores):
            length = int(response_lengths[index].item())
            rewards[index, length - 1] = score["overall"]
            for key, value in score.items():
                metrics[key].append(value)

        return rewards, metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)

        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as error:
            raise RuntimeError(f"Failed to load reward function: {error}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        function = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")

        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")

        self.reward_fn = partial(function, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")