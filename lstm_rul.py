# pip install torch numpy  — LSTM으로 남은 수명(RUL) 예측 예제 (가짜 데이터, 미실행)
import numpy as np
import torch
import torch.nn as nn

def make_robot(life=200):
    t = np.arange(life)
    wear = (t / life) ** 2
    vib  = 0.1 + 0.9 * wear + np.random.normal(0, 0.03, life)
    cur  = 1.8 + 0.6 * wear + np.random.normal(0, 0.02, life)
    temp = 38  + 10  * wear + np.random.normal(0, 0.3, life)
    X = np.stack([vib, cur, temp], axis=1)
    rul = (life - t) / life
    return X, rul

SEQ = 20

def to_windows(X, rul):
    xs, ys = [], []
    for i in range(len(X) - SEQ):
        xs.append(X[i:i + SEQ]); ys.append(rul[i + SEQ])
    return np.array(xs), np.array(ys)

xs, ys = [], []
for _ in range(30):
    X, r = make_robot(np.random.randint(150, 250))
    a, b = to_windows(X, r); xs.append(a); ys.append(b)
X = np.concatenate(xs); y = np.concatenate(ys)
mean, std = X.reshape(-1, 3).mean(0), X.reshape(-1, 3).std(0)
X = (X - mean) / std
idx = np.random.permutation(len(X)); cut = int(len(X) * 0.8)
X_tr, y_tr = torch.tensor(X[idx[:cut]], dtype=torch.float32), torch.tensor(y[idx[:cut]], dtype=torch.float32)
X_te, y_te = torch.tensor(X[idx[cut:]], dtype=torch.float32), torch.tensor(y[idx[cut:]], dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)

model = LSTMModel()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
for epoch in range(30):
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), 64):
        b = perm[i:i + 64]
        opt.zero_grad(); loss = loss_fn(model(X_tr[b]), y_tr[b]); loss.backward(); opt.step()
    if epoch % 5 == 0:
        with torch.no_grad():
            print(f"epoch {epoch:2d}  테스트 오차(MSE): {loss_fn(model(X_te), y_te).item():.4f}")
with torch.no_grad():
    pred = model(X_te[:5])
for p, t in zip(pred, y_te[:5]):
    print(f"예측 남은 수명: {p:.0%}   실제: {t:.0%}")
