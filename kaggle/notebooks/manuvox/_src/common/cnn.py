class TemporalCNN(nn.Module):
    """Coordinates + velocity -> 6 depthwise-separable residual conv blocks -> mean over time -> head."""

    def __init__(self, num_classes=250, channels=192, blocks=6, kernel=5, dropout=0.2):
        super().__init__()
        self.features = SimpleFeatures()
        self.proj = nn.Linear(self.features.output_dim, channels)
        self.blocks = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, 1),
                    nn.BatchNorm1d(channels),
                    nn.Dropout(dropout),
                )
                for _ in range(blocks)
            )
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(channels, num_classes))

    def set_progress(self, epoch, epochs):
        pass

    def forward(self, x):
        h = self.proj(self.features(x)).transpose(1, 2)
        for block in self.blocks:
            h = F.gelu(h + block(h))
        return self.head(h.mean(dim=-1))
