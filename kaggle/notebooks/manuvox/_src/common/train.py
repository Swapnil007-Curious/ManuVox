import copy


def seed_torch(seed=SEED):
    """Seed torch on every device. GPU kernels can still differ bitwise between runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class EMA:
    """A frozen copy whose weights follow ema = decay * ema + (1 - decay) * live after each step."""

    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for avg, live in zip(self.module.parameters(), model.parameters()):
            avg.lerp_(live.detach(), 1.0 - self.decay)
        for stored, current in zip(self.module.buffers(), model.buffers()):
            stored.copy_(current)


def warmup_cosine(total_steps, warmup_fraction=0.05):
    """LR multiplier per step: linear warmup, then cosine decay to zero."""
    warmup = max(1, round(total_steps * warmup_fraction))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    return factor


@torch.no_grad()
def predict_logits(model, x, batch_size=1024):
    """Float32 logits in eval mode, in batches."""
    model.eval()
    return torch.cat([model(x[i : i + batch_size].float()).float() for i in range(0, len(x), batch_size)])


def top_k(logits, y, k=1):
    """Share of rows whose label is among the k highest logits."""
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).float().mean().item()


def train_model(model, x_train, y_train, x_val=None, y_val=None, *, epochs, lr, recipe,
                batch_size=256, weight_decay=0.05, label_smoothing=0.1, ema_decay=None,
                patience=None, seed=SEED, quiet=False):
    """Train with AdamW, warmup + cosine LR and mixed precision on GPU.

    With validation data, the epoch with the best val top-1 is kept (EMA weights when
    ``ema_decay`` is set); without it, the last epoch is kept. Returns (model, history).
    """
    seed_torch(seed)
    model = model.to(DEVICE)
    fit_standardisation(model.features, x_train)
    ema = EMA(model, ema_decay) if ema_decay else None
    scored = ema.module if ema else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(epochs * steps_per_epoch))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    order_gen = torch.Generator().manual_seed(seed)
    augment_gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    history, best, best_state, stale = [], -1.0, None, 0
    report_every = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        model.set_progress(epoch, epochs)
        order = torch.randperm(len(x_train), generator=order_gen).to(DEVICE)
        loss_sum, correct = torch.zeros((), device=DEVICE), torch.zeros((), device=DEVICE)
        for i in range(0, len(order), batch_size):
            index = order[i : i + batch_size]
            batch, target = augment_batch(x_train[index].float(), recipe, augment_gen), y_train[index]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
                loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)
            loss_sum += loss.detach().float() * len(index)
            correct += (logits.argmax(1) == target).sum()
        row = {"epoch": epoch, "train_loss": loss_sum.item() / len(order), "train_top1": correct.item() / len(order)}
        if x_val is not None:
            row["val_top1"] = top_k(predict_logits(scored, x_val), y_val)
        row["seconds"] = time.perf_counter() - started
        history.append(row)
        improved = x_val is None or row["val_top1"] > best
        if improved:
            best, stale = row.get("val_top1", best), 0
            best_state = copy.deepcopy(scored.state_dict())
        else:
            stale += 1
        if not quiet and (epoch % report_every == 0 or epoch == 1 or epoch == epochs):
            val = f", val top-1 {row['val_top1']:.3f}" if x_val is not None else ""
            print(f"epoch {epoch:3d}/{epochs}: train loss {row['train_loss']:.3f}, "
                  f"train top-1 {row['train_top1']:.3f}{val} ({row['seconds']:.1f} s)")
        if patience and stale >= patience:
            print(f"early stop after epoch {epoch}: no val gain for {patience} epochs")
            break
    scored.load_state_dict(best_state)
    return scored.eval(), pd.DataFrame(history)


def bootstrap_ci(values, n_boot=1000, seed=SEED):
    """95% percentile interval of the mean of ``values`` (1-D), resampling rows."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])
