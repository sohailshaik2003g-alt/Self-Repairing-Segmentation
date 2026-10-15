"""
Neural Cellular Automata (NCA) model for 2D segmentation.

Architecture:
  - Each "cell" in a 2D grid holds a state vector of `n_channels` dimensions.
  - At each step, a shared local update rule is applied:
      1. PERCEPTION: Efficient depthwise 3x3 convolutions to sense neighbors:
         identity + Sobel-x + Sobel-y per channel = 3 * total_channels features.
      2. UPDATE: A small per-cell MLP (two 1x1 convolutions with ReLU) maps the
         perceived features to a state delta.
      3. STOCHASTIC UPDATE MASK: Each cell fires with probability `fire_rate`,
         simulating asynchronous local updates.
      4. RESIDUAL & CLAMP: new_state = clamp(old_state + delta * fire_mask, -10, 10).
  - Channel 0 of the hidden state is the raw segmentation logit.
  - The input image is injected as a fixed channel concatenated with hidden state.

Fits comfortably in 4 GB VRAM (GTX 1650).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class NCAModel(nn.Module):
    """Neural Cellular Automata for 2D medical image segmentation."""

    def __init__(
        self,
        n_channels: int = 16,
        hidden_dim: int = 64,
        fire_rate: float = 0.5,
        input_channels: int = 1,
    ):
        """
        Args:
            n_channels: number of state channels per cell (channel 0 is seg logit).
            hidden_dim: hidden dimension of the per-cell MLP.
            fire_rate: probability each cell updates at each step.
            input_channels: number of input image channels (1 for grayscale).
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.fire_rate = fire_rate
        self.input_channels = input_channels

        # Total state channels = input (fixed) + hidden (evolving)
        self.total_channels = input_channels + n_channels

        # --- Perception filters (depthwise: 3 filters per channel) ---
        self.perception_channels = 3 * self.total_channels
        self._build_perception_filters()

        # --- Update MLP (1x1 Conv -> ReLU -> 1x1 Conv) ---
        self.update_net = nn.Sequential(
            nn.Conv2d(self.perception_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_channels, kernel_size=1, bias=False),
        )

        # Initialize last layer to near-zero so initial updates are gentle
        nn.init.zeros_(self.update_net[-1].weight)

    def _build_perception_filters(self):
        """Build fixed depthwise 3x3 perception filters (Identity, Sobel-X, Sobel-Y)."""
        ident = torch.tensor([[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32) / 8.0

        C = self.total_channels
        # Shape: (3*C, 1, 3, 3) for depthwise convolution with groups=C
        # For each channel c in 0..C-1, we produce 3 output channels: [ident, sobel_x, sobel_y]
        filters = torch.stack([ident, sobel_x, sobel_y], dim=0)  # (3, 3, 3)
        weight = filters.repeat(C, 1, 1).unsqueeze(1)  # (3*C, 1, 3, 3)
        self.register_buffer("perception_kernel", weight)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Apply perception filters via depthwise convolution.

        Args:
            state: (B, total_channels, H, W)
        Returns:
            perceived: (B, 3*total_channels, H, W)
        """
        return F.conv2d(
            state,
            self.perception_kernel,
            padding=1,
            groups=self.total_channels,
        )

    def update(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state delta via per-cell MLP."""
        perceived = self.perceive(state)
        delta = self.update_net(perceived)
        return delta

    def get_fire_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate stochastic firing mask."""
        if self.fire_rate >= 1.0:
            return torch.ones(B, 1, H, W, device=device)
        return (torch.rand(B, 1, H, W, device=device) < self.fire_rate).float()

    def forward(
        self,
        image: torch.Tensor,
        n_steps: int = 32,
        return_all_steps: bool = False,
        initial_state: Optional[torch.Tensor] = None,
        fire_mask_override: Optional[float] = None,
    ):
        """Run the NCA for n_steps.

        Args:
            image: (B, input_channels, H, W) in [0, 1]
            n_steps: number of NCA steps
            return_all_steps: whether to return intermediate hidden states
            initial_state: optional initial hidden state (B, n_channels, H, W)
            fire_mask_override: if provided, use this fixed scalar instead of random mask

        Returns:
            seg_logits: (B, 1, H, W)
            (optional) all_states: list of (B, n_channels, H, W)
        """
        B, _, H, W = image.shape
        device = image.device

        if initial_state is not None:
            hidden = initial_state
        else:
            hidden = torch.zeros(B, self.n_channels, H, W, device=device)

        all_states = [hidden.clone()] if return_all_steps else None

        for step in range(n_steps):
            state = torch.cat([image, hidden], dim=1)
            delta = self.update(state)

            if fire_mask_override is not None:
                mask = fire_mask_override
            else:
                mask = self.get_fire_mask(B, H, W, device)

            # Residual update with clamping to ensure numerical stability
            hidden = torch.clamp(hidden + delta * mask, -10.0, 10.0)

            if return_all_steps:
                all_states.append(hidden.clone())

        seg_logits = hidden[:, 0:1, :, :]

        if return_all_steps:
            return seg_logits, all_states
        return seg_logits

    def forward_with_damage(
        self,
        image: torch.Tensor,
        n_steps: int = 32,
        damage_fn=None,
        damage_step: Optional[int] = None,
        return_all_steps: bool = True,
    ):
        """Run NCA with a damage event at damage_step.

        Args:
            image: (B, input_channels, H, W)
            n_steps: total number of steps
            damage_fn: callable(hidden) -> damaged_hidden
            damage_step: step index (0 to n_steps-1) at which damage occurs
            return_all_steps: whether to save states at each step

        Returns:
            seg_logits: (B, 1, H, W)
            all_states: list of (B, n_channels, H, W)
        """
        B, _, H, W = image.shape
        device = image.device
        hidden = torch.zeros(B, self.n_channels, H, W, device=device)

        all_states = [hidden.clone()] if return_all_steps else None

        for step in range(n_steps):
            # Apply damage if this is the damage step
            if damage_fn is not None and step == damage_step:
                hidden = damage_fn(hidden)
                if return_all_steps:
                    all_states.append(hidden.clone())  # record damaged state

            state = torch.cat([image, hidden], dim=1)
            delta = self.update(state)
            mask = self.get_fire_mask(B, H, W, device)

            hidden = torch.clamp(hidden + delta * mask, -10.0, 10.0)

            if return_all_steps:
                all_states.append(hidden.clone())

        seg_logits = hidden[:, 0:1, :, :]

        if return_all_steps:
            return seg_logits, all_states
        return seg_logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = NCAModel(16, 64, 0.5)
    print("Params:", count_parameters(m))
    x = torch.randn(2, 1, 64, 64)
    out = m(x, n_steps=16)
    print("Output:", out.shape, "range:", out.min().item(), out.max().item())
