import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
import os

# ============================================================
# 1. CLASE DE CONFIGURACIÓN (definida PRIMERO)
# ============================================================
class Config:
    batch_size = 16
    max_iters = 3000
    eval_interval = 100
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    vocab_size = 128
    block_size = 15
    n_embd = 64
    n_head = 2
    n_layer = 2
    dropout = 0.0 
    
config_name = "Main"

# ============================================================
# 2. SELECCIÓN DE CONFIGURACIÓN (AHORA Config YA EXISTE)
# ============================================================
# Descomenta UNA de las siguientes opciones y comenta las otras dos.

# ---------- Configuración A: Dropout (regularización) ----------
# config_name = "A_dropout"
# Config.block_size = 15
# Config.n_embd = 64
# Config.n_layer = 2
# Config.dropout = 0.1

# ---------- Configuración B: Más capas (profundidad) ----------
# config_name = "B_deeper"
# Config.block_size = 15
# Config.n_embd = 64
# Config.n_layer = 3
# Config.dropout = 0.0

# ---------- Configuración C: Más capas + Dropout ----------
# config_name = "C_deeper_dropout"
# Config.block_size = 15
# Config.n_embd = 64
# Config.n_layer = 3
# Config.dropout = 0.1

# Nota: la configuración principal (block_size=15, n_embd=64, n_layer=2, dropout=0.0)
# ya está entrenada anteriormente, por lo que no la repetimos aquí.


# ============================================================
# 3. CARGA DE DATOS
# ============================================================
state_codes_path = "../../notebooks/state_codes_all.pkl"

with open(state_codes_path, 'rb') as f:
    state_codes = pickle.load(f)

data = torch.tensor(state_codes, dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Datos cargados: {len(data)} estados")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
print(f"Configuración: {config_name}")
print(f"block_size={Config.block_size}, n_embd={Config.n_embd}, n_layer={Config.n_layer}, dropout={Config.dropout}")


# ============================================================
# 4. MODELO
# ============================================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(Config.vocab_size, Config.n_embd)
        self.position_embedding_table = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[Block(Config.n_embd, Config.n_head) for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, Config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=Config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# 5. FUNCIONES DE UTILIDAD
# ============================================================
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i+Config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
    x, y = x.to(Config.device), y.to(Config.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ============================================================
# 6. ENTRENAMIENTO
# ============================================================
model = GPTLanguageModel()
m = model.to(Config.device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

# Listas para almacenar pérdidas
train_losses = []
val_losses = []
steps = []

for iter in range(Config.max_iters):
    if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        steps.append(iter)

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ============================================================
# 7. EVALUACIÓN EN TEST
# ============================================================
print("\n--- Evaluación en test (estado completo) ---")

def evaluate_full_state(model, data, block_size):
    model.eval()
    total_correct = 0
    total_samples = 0
    per_network_correct = torch.zeros(7)
    per_network_total = torch.zeros(7)

    with torch.no_grad():
        for i in range(len(data) - block_size):
            context = data[i:i+block_size].unsqueeze(0).to(Config.device)
            target = data[i+block_size].item()

            logits, _ = model(context)
            pred = logits[0, -1, :].argmax().item()

            if pred == target:
                total_correct += 1
            total_samples += 1

            target_bits = [(target >> b) & 1 for b in range(6, -1, -1)]
            pred_bits = [(pred >> b) & 1 for b in range(6, -1, -1)]
            for b in range(7):
                if target_bits[b] == pred_bits[b]:
                    per_network_correct[b] += 1
                per_network_total[b] += 1

    exact_acc = total_correct / total_samples
    per_network_acc = per_network_correct / per_network_total
    return exact_acc, per_network_acc

exact_acc, per_network_acc = evaluate_full_state(model, val_data, Config.block_size)
print(f"Exact-match accuracy: {exact_acc:.4f}")
print("Per-network accuracy:")
for i, acc in enumerate(per_network_acc, 1):
    print(f"  Red {i}: {acc:.4f}")

def compute_per_network_f1(model, data, block_size):
    model.eval()
    tp = torch.zeros(7)
    fp = torch.zeros(7)
    fn = torch.zeros(7)

    with torch.no_grad():
        for i in range(len(data) - block_size):
            context = data[i:i+block_size].unsqueeze(0).to(Config.device)
            target = data[i+block_size].item()

            logits, _ = model(context)
            pred = logits[0, -1, :].argmax().item()

            target_bits = [(target >> b) & 1 for b in range(6, -1, -1)]
            pred_bits = [(pred >> b) & 1 for b in range(6, -1, -1)]
            for b in range(7):
                if target_bits[b] == 1 and pred_bits[b] == 1:
                    tp[b] += 1
                elif target_bits[b] == 0 and pred_bits[b] == 1:
                    fp[b] += 1
                elif target_bits[b] == 1 and pred_bits[b] == 0:
                    fn[b] += 1

    f1 = torch.zeros(7)
    for b in range(7):
        if tp[b] + fp[b] > 0 and tp[b] + fn[b] > 0:
            precision = tp[b] / (tp[b] + fp[b])
            recall = tp[b] / (tp[b] + fn[b])
            f1[b] = 2 * precision * recall / (precision + recall)
    return f1

f1_per_network = compute_per_network_f1(model, val_data, Config.block_size)
print("\nF1-score por red:")
for i, f1 in enumerate(f1_per_network, 1):
    print(f"  Red {i}: {f1:.4f}")

print("\n--- Evaluación en test (cambios) ---")

def evaluate_changes(model, data, block_size):
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for i in range(len(data) - block_size - 1):
            context = data[i:i+block_size].unsqueeze(0).to(Config.device)
            current = data[i+block_size].item()
            next_token = data[i+block_size+1].item()
            target_change = 1 if current != next_token else 0

            logits, _ = model(context)
            pred_token = logits[0, -1, :].argmax().item()
            pred_change = 1 if pred_token != current else 0

            if target_change == 1 and pred_change == 1:
                tp += 1
            elif target_change == 0 and pred_change == 1:
                fp += 1
            elif target_change == 1 and pred_change == 0:
                fn += 1
            else:
                tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

acc_changes, prec_changes, rec_changes, f1_changes = evaluate_changes(model, val_data, Config.block_size)
print(f"Accuracy (cambios): {acc_changes:.4f}")
print(f"Precision (cambios): {prec_changes:.4f}")
print(f"Recall (cambios): {rec_changes:.4f}")
print(f"F1-score (cambios): {f1_changes:.4f}")

print("\n--- Secuencia generada (primeros 20 tokens) ---")
context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
generated = model.generate(context, max_new_tokens=20)[0].tolist()
print(generated)


# ============================================================
# 8. GUARDAR RESULTADOS
# ============================================================
results = {
    'config_name': config_name,
    'config': {
        'block_size': Config.block_size,
        'n_embd': Config.n_embd,
        'n_layer': Config.n_layer,
        'n_head': Config.n_head,
        'dropout': Config.dropout,
        'batch_size': Config.batch_size,
        'learning_rate': Config.learning_rate,
        'max_iters': Config.max_iters,
    },
    'train_losses': train_losses,
    'val_losses': val_losses,
    'steps': steps,
    'exact_acc': exact_acc,
    'per_network_acc': per_network_acc.tolist(),
    'f1_per_network': f1_per_network.tolist(),
    'f1_changes': f1_changes,
    'accuracy_changes': acc_changes,
    'precision_changes': prec_changes,
    'recall_changes': rec_changes,
    'generated': generated,
}

filename = f"results_{config_name}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResultados guardados en {filename}")

# Guardar modelo
torch.save(model.state_dict(), f'gpt_brain_model_{config_name}.pth')
print(f"Modelo guardado en gpt_brain_model_{config_name}.pth")