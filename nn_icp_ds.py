import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressivePositionalEncoding(nn.Module):
    def __init__(self, max_bands, include_input=True):
        super().__init__()
        self.max_bands = max_bands
        self.include_input = include_input
        self.register_buffer("current_band", torch.tensor(0, dtype=torch.int))

    def forward(self, x):
        encodings = []
        if self.include_input:
            encodings.append(x)

        for i in range(self.max_bands):
            freq = 2**i
            sin = torch.sin(freq * torch.pi * x)
            cos = torch.cos(freq * torch.pi * x)
            mask = (i < self.current_band).float()
            sin = sin * mask
            cos = cos * mask
            encodings.append(sin)
            encodings.append(cos)

        return torch.cat(encodings, dim=-1)

    def set_current_band(self, current_band):
        self.current_band.fill_(current_band)


def axis_angle_to_rotation(omega):
    theta = torch.norm(omega, dim=1, keepdim=True)
    epsilon = 1e-7
    theta = torch.clamp(theta, min=epsilon)
    axis = omega / theta

    K = torch.zeros((omega.size(0), 3, 3), device=omega.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=omega.device).unsqueeze(0).repeat(omega.size(0), 1, 1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)

    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    small_theta_mask = theta.squeeze() < epsilon
    if small_theta_mask.any():
        K_small = K[small_theta_mask]
        R_small = I[small_theta_mask] + K_small + 0.5 * (K_small @ K_small)
        R[small_theta_mask] = R_small

    return R


class ModifiedICPModel(nn.Module):
    def __init__(self, max_bands=5, hidden_dim=256, include_input=True):
        super().__init__()
        self.pos_encoder = ProgressivePositionalEncoding(max_bands, include_input)
        input_dim = 3 * (1 + 2 * max_bands) if include_input else 3 * 2 * max_bands
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, x):
        x_enc = self.pos_encoder(x)
        out = self.mlp(x_enc)
        axis_angle = out[:, :3]
        translation = out[:, 3:]
        return axis_angle, translation


def chamfer_distance(source, target):
    dist_matrix = torch.cdist(source, target)
    min_dist_src_to_tgt = torch.min(dist_matrix, dim=1)[0]
    min_dist_tgt_to_src = torch.min(dist_matrix, dim=0)[0]
    chamfer_loss = (min_dist_src_to_tgt.mean() + min_dist_tgt_to_src.mean()) / 2
    return chamfer_loss


# Example usage:
model = ModifiedICPModel(max_bands=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

num_epochs = 200
max_bands = 5
epochs_per_band = 40  # Increase band every 40 epochs

for epoch in range(num_epochs):
    current_band = min(max_bands, (epoch // epochs_per_band))
    model.pos_encoder.set_current_band(current_band)

    for source, target in dataloader:  # Assume dataloader provides source-target pairs
        optimizer.zero_grad()
        axis_angle, translation = model(source)
        R = axis_angle_to_rotation(axis_angle)
        transformed = torch.bmm(R, source.unsqueeze(-1)).squeeze(-1) + translation
        loss = chamfer_distance(transformed, target)
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}, Current Band: {current_band}")
