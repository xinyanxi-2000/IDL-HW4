import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        # TODO: Implement create_pe_table
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        return x + self.pe[:, :seq_len]


class RotaryPositionalEncoding(nn.Module):
    """Apply rotary position information to attention queries and keys."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0):
        """
        Initialize the RotaryPositionalEncoding.
        Args:
            head_dim (int): The dimension of each attention head.
            max_len (int): The maximum context length.
            base (float): The base used for the rotary frequencies.
        """
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary encoding")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim = head_dim
        self.max_len = max_len

        # TODO: Create one inverse frequency for each neighboring feature pair.
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float)  # 0, 2, 4, ...
        inverse_frequencies = 1.0 / (base ** (dimensions / head_dim))  # (head_dim/2,)

        # TODO: Build the angle table for every supported position, then
        # register its cosine and sine as non-persistent buffers named
        # ``cos`` and ``sin``.
        positions = torch.arange(max_len, dtype=torch.float)  # (max_len,)
        angles = torch.outer(positions, inverse_frequencies)  # (max_len, head_dim/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass for the RotaryPositionalEncoding.
        Args:
            x (torch.Tensor): Split attention heads. shape: (B, H, T, head_dim)
            start_pos (int): Absolute position of the first token in x.
        Returns:
            torch.Tensor: Rotated attention heads with the same shape as x.
        """
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, heads, sequence, head_dim)")
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, but received {x.size(-1)}"
            )
        if not isinstance(start_pos, int) or isinstance(start_pos, bool):
            raise TypeError("start_pos must be an integer")
        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        # DO NOT MODIFY: input and context validation.
        seq_len = x.size(-2)
        end_pos = start_pos + seq_len
        if end_pos > self.max_len:
            raise ValueError(
                f"Positions up to {end_pos - 1} exceed max_len {self.max_len}"
            )

        # TODO: Select the cosine and sine rows beginning at start_pos and
        # cast them to the input dtype.
        cos = self.cos[start_pos:end_pos].to(x.dtype)  # (T, head_dim/2)
        sin = self.sin[start_pos:end_pos].to(x.dtype)

        # TODO: Rotate every neighboring even/odd feature pair and interleave
        # the results so the output has the same shape as x.
        x_even = x[..., 0::2]  # (B, H, T, head_dim/2)
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        output = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)  # re-interleave
        return output
