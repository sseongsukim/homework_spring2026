"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        input_dim = state_dim
        output_dim = chunk_size * action_dim  # (Batch_size, chunk_size * action_dim)

        # MLP layers
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i, (prev_dim, dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_size, action_dim = action_chunk.shape
        actions = action_chunk.reshape(batch_size, -1)
        pred = self.layers(state)
        loss = torch.nn.functional.mse_loss(pred, actions)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred_actions = self.layers(state)
        pred_action_chunk = pred_actions.reshape(-1, self.chunk_size, self.action_dim)
        return pred_action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        noise_dim = 1
        output_dim = chunk_size * action_dim  # (Batch_size, chunk_size * action_dim)
        input_dim = state_dim + output_dim + noise_dim

        # MLP layers
        layers = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        for i, (prev_dim, dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_size, action_dim = action_chunk.shape
        actions = action_chunk.reshape(batch_size, -1)
        device = action_chunk.device

        x_0 = torch.randn(
            size=(batch_size, chunk_size * action_dim),
            dtype=torch.float32,
            device=device,
        )
        x_1 = actions
        t = torch.rand(size=(batch_size, 1), device=device)

        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.layers(torch.concat([state, x_t, t], dim=-1))
        loss = torch.nn.functional.mse_loss(pred, vel)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        batch_size, _ = state.shape
        device = state.device

        actions = torch.randn(
            size=(batch_size, self.chunk_size * self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        for i in range(num_steps):
            t = torch.full(
                size=(batch_size, 1), fill_value=i / num_steps, device=device
            )
            vel = self.layers(torch.concat([state, actions, t], dim=-1))
            actions += vel / num_steps

        return actions.reshape(batch_size, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
