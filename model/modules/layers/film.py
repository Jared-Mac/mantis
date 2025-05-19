import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Applies affine transformation (scale and shift) to feature maps.
    """
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature maps of shape (B, C, H, W) or (B, C).
            gamma (torch.Tensor): Scale parameters of shape (B, C) or (B, C, 1, 1) for 2D.
            beta (torch.Tensor): Shift parameters of shape (B, C) or (B, C, 1, 1) for 2D.

        Returns:
            torch.Tensor: Modulated feature maps.
        """
        # Ensure gamma and beta are broadcastable to x's shape
        if x.dim() == 4 and gamma.dim() == 2: # (B,C) for (B,C,H,W)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2 and gamma.dim() == 4: # (B,C,1,1) for (B,C) - less common but handle
             gamma = gamma.squeeze(-1).squeeze(-1)
             beta = beta.squeeze(-1).squeeze(-1)
        elif x.dim() != gamma.dim():
            raise ValueError(f"Input x dim {x.dim()} and gamma/beta dim {gamma.dim()} are not compatible for broadcasting.")

        return gamma * x + beta

class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma and beta) from a conditioning input.
    """
    def __init__(self, cond_dim: int, num_features: int, hidden_dim: int = None):
        """
        Args:
            cond_dim (int): Dimensionality of the conditioning input.
            num_features (int): Number of features to modulate (e.g., channels in a Conv layer).
            hidden_dim (int, optional): Dimension of the hidden layer.
                                       If None, uses cond_dim. Defaults to None.
        """
        super().__init__()
        self.num_features = num_features
        if hidden_dim is None:
            hidden_dim = cond_dim

        self.gamma_beta_generator = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_features) # 2 for gamma and beta
        )

    def forward(self, cond_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond_input (torch.Tensor): Conditioning input of shape (B, cond_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                gamma (torch.Tensor): Scale parameters of shape (B, num_features).
                beta (torch.Tensor): Shift parameters of shape (B, num_features).
        """
        gamma_beta = self.gamma_beta_generator(cond_input)
        gamma = gamma_beta[:, :self.num_features]
        beta = gamma_beta[:, self.num_features:]
        return gamma, beta

if __name__ == '__main__':
    # Example Usage
    B, C_feat, H, W = 4, 64, 32, 32
    B, C_cond = 4, 128

    # Input features
    features = torch.randn(B, C_feat, H, W)
    features_flat = torch.randn(B, C_feat)

    # Conditioning input
    conditioning_signal = torch.randn(B, C_cond)

    # FiLM Generator
    film_gen = FiLMGenerator(cond_dim=C_cond, num_features=C_feat, hidden_dim=256)
    gamma, beta = film_gen(conditioning_signal)
    print(f"Generated gamma shape: {gamma.shape}, beta shape: {beta.shape}") # Expected: (B, C_feat)

    # FiLM Layer
    film_layer = FiLMLayer()

    # Apply to 2D features (e.g., after a fully connected layer)
    modulated_features_flat = film_layer(features_flat, gamma, beta)
    print(f"Modulated flat features shape: {modulated_features_flat.shape}") # Expected: (B, C_feat)

    # Apply to 4D features (e.g., convolutional feature maps)
    modulated_features_conv = film_layer(features, gamma, beta) # gamma/beta will be unsqueezed
    print(f"Modulated conv features shape: {modulated_features_conv.shape}") # Expected: (B, C_feat, H, W)

    # Test with pre-shaped gamma/beta for 4D features
    gamma_4d = gamma.unsqueeze(-1).unsqueeze(-1)
    beta_4d = beta.unsqueeze(-1).unsqueeze(-1)
    modulated_features_conv_preshaped = film_layer(features, gamma_4d, beta_4d)
    print(f"Modulated conv features (preshaped gamma/beta) shape: {modulated_features_conv_preshaped.shape}")

    # Example with FiLMGenerator for a different number of features
    C_feat_2 = 128
    film_gen_2 = FiLMGenerator(cond_dim=C_cond, num_features=C_feat_2)
    gamma_2, beta_2 = film_gen_2(conditioning_signal)
    features_2 = torch.randn(B, C_feat_2, H // 2, W // 2)
    modulated_features_2 = film_layer(features_2, gamma_2, beta_2)
    print(f"Modulated features_2 shape: {modulated_features_2.shape}")

    print("FiLM modules created and example usage demonstrated successfully.")
