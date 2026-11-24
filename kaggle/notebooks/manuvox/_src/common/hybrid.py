class SelfAttention(nn.Module):
    """Multi-head self-attention written with Linear and MatMul, which export and quantise cleanly."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.query, self.key, self.value, self.out = (nn.Linear(dim, dim) for _ in range(4))
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        return x.reshape(-1, x.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        q = self.split(self.query(x)) * self.head_dim**-0.5
        weights = torch.softmax(q @ self.split(self.key(x)).transpose(-2, -1), dim=-1)
        context = self.dropout(weights) @ self.split(self.value(x))
        return self.out(context.transpose(1, 2).flatten(2))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim, heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff =nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.ff(self.norm2(x)))


class ConvBlock(nn.Module):
    """Depthwise temporal conv -> BatchNorm -> SiLU -> expand -> SiLU -> project, residual. (B, C, T)."""

    def __init__(self, dim, kernel=17, expansion=512, dropout=0.2):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.expand, self.project = nn.Conv1d(dim, expansion, 1), nn.Conv1d(expansion, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.norm(self.depthwise(x)))
        return x + self.dropout(self.project(F.silu(self.expand(h))))


class Stage(nn.Module):
    """Two conv blocks over time, then one transformer block. (B, T, C)."""

    def __init__(self, dim):
        super().__init__()
        self.convs = nn.Sequential(ConvBlock(dim), ConvBlock(dim))
        self.transformer = TransformerBlock(dim)

    def forward(self, x):
        return self.transformer(self.convs(x.transpose(1, 2)).transpose(1, 2))


class HybridClassifier(nn.Module):
    """Features -> linear projection -> 3 conv+transformer stages -> mean over time -> head.

    Head dropout switches on halfway through training ("late dropout").
    """

    def __init__(self, num_classes=250, dim=256, stages=3, head_dropout=0.4):
        super().__init__()
        self.features = FeatureBlock()
        self.proj = nn.Linear(self.features.output_dim, dim)
        self.stages = nn.Sequential(*(Stage(dim) for _ in range(stages)))
        self.head_dropout, self.late_dropout_on = head_dropout, False
        self.head = nn.Linear(dim, num_classes)

    def set_progress(self, epoch, epochs):
        """Called before each 1-based epoch: head dropout is on for the second half."""
        self.late_dropout_on = (epoch - 1) >= 0.5 * epochs

    def forward(self, x):
        h = self.stages(self.proj(self.features(x))).mean(dim=1)
        if self.training and self.late_dropout_on:
            h = F.dropout(h, self.head_dropout)
        return self.head(h)
