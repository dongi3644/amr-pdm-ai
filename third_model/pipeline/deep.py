"""딥러닝 모델 (1D-CNN): 1초 파형을 그대로 넣는 감지 모델과 원인 분류 모델

[역할] 사람이 정한 특징(진동 rms·kurt …) 대신 1초 동안의 파형 자체를 신경망에 넣는다.
    ① CNNAEDetector     : 1D-CNN 오토인코더. 정상 파형만 학습 → 복원 오차 = 이상 점수 (학습에 없는 고장도 감지)
    ② DeepCauseClassifier: 1D-CNN(파형) + 작은 신경망(특징) → 원인(종류|바퀴) 확률. 판단 근거는 입력×기울기(Grad×Input)

[입력] 1초 창마다 채널 × WAVE_LEN(기본 512) 배열
    가속도 ax·ay·az (config.VIB_CHANNELS) + 학습에 쓴 엔코더의 바퀴 속도 파형 (예: spd_fr)
    로봇마다 처음 5분(per_unit) 또는 공통 기준(reference)의 채널별 평균·표준편차로 정규화한다 (WaveBaseline).

[기존 코드와 연결] 모델이 받는 Z = [특징 z (기존과 같음) | 정규화한 파형을 펼친 값]
    기존 특징 모델(z-score·ECOD·PCA·오토인코더·랜덤포레스트)은 SlicedDetector·SlicedCause가 앞쪽 특징 열만 잘라 넘긴다.
    그래서 건강 점수 눈금(3·6·9), 1분 판정, 교차검증, 보고서는 그대로 쓴다.

[필요한 것] PyTorch (CPU로 충분):  pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
import numpy as np

import config as C
from pipeline.models import Z_CLIP, SegmentBaseline

WHEELS = ["fl", "fr", "rl", "rr"]
WHEEL_KO = {"fl": "앞 왼쪽", "fr": "앞 오른쪽", "rl": "뒤 왼쪽", "rr": "뒤 오른쪽"}


# ------------------------------------------------------------ 파형 만들기
def wave_channels(feats):
    """학습에 쓰는 특징 목록 → 파형 채널 목록. 가속도 축 전부 + 특징에 들어간 엔코더 바퀴의 속도 파형"""
    used = [w for w in WHEELS if any(f.startswith(w + "_") for f in feats)]
    return list(C.VIB_CHANNELS) + [f"spd_{w}" for w in used]


def wave_from_rec(rec, L=None):
    """1초 창(dataio rec) → {채널: 길이 L 배열}. 진동은 같은 간격으로 다시 뽑고, 엔코더는 누적 펄스를 미분해 속도 파형으로"""
    L = L or C.DEEP_WAVE_LEN
    out, grid = {}, np.linspace(0.0, 1.0, L, endpoint=False)
    for ch in C.VIB_CHANNELS:
        v = np.asarray(rec[ch], float)
        if len(v) < 2:
            out[ch] = np.zeros(L, np.float32)
            continue
        x = np.linspace(0.0, 1.0, len(v), endpoint=False)
        out[ch] = np.interp(grid, x, v).astype(np.float32)
    for k, w in zip(C.ENCODER_CHANNELS, WHEELS):
        t, c = rec.get(f"{k}_t"), rec.get(f"{k}_c")
        if t is None or c is None or len(t) < 3:
            continue
        cum = np.interp(grid, np.asarray(t, float), np.asarray(c, float))
        sp = np.abs(np.gradient(cum) * L)                   # 펄스/초
        k_ = 25                                             # 약 50 ms 이동 평균 (펄스가 띄엄띄엄 오는 계단을 부드럽게)
        sp = np.convolve(sp, np.ones(k_) / k_, mode="same")
        out[f"spd_{w}"] = sp.astype(np.float32)
    return out


def wave_ko(ch):
    if ch.startswith("spd_"):
        return f"{WHEEL_KO.get(ch[4:], ch[4:])} 바퀴 속도 파형"
    return f"{C.CHANNEL_KO.get(ch, ch)} 파형"


class WaveBaseline(SegmentBaseline):
    """기존 특징 기준(구간별 z) + 파형 채널별 평균·표준편차. z()는 [특징 z | 정규화 파형(펼침)]을 돌려준다."""

    def __init__(self, wave_ch):
        self.wave_ch = list(wave_ch)

    def fit(self, rows, feats=None):
        super().fit(rows, feats)
        use = [r for r in rows if not r["bump"]]
        mu, sd = [], []
        for ch in self.wave_ch:
            A = np.concatenate([r["_wave"].get(ch, np.zeros(C.DEEP_WAVE_LEN, np.float32)) for r in use])
            m, s = float(A.mean()), float(A.std())
            mu.append(m)
            sd.append(max(s, 1e-3 + 0.05 * abs(m)))
        self.wmu, self.wsd = np.array(mu)[:, None], np.array(sd)[:, None]
        return self

    def waves(self, rows):
        L = C.DEEP_WAVE_LEN
        W = np.stack([np.stack([r["_wave"].get(ch, np.zeros(L, np.float32)) for ch in self.wave_ch]) for r in rows])
        return np.clip((W - self.wmu) / self.wsd, -Z_CLIP, Z_CLIP).astype(np.float32)

    def z(self, rows):
        Zf = super().z(rows)
        if not len(rows):
            return np.empty((0, Zf.shape[1] + len(self.wave_ch) * C.DEEP_WAVE_LEN))
        return np.hstack([Zf, self.waves(rows).reshape(len(rows), -1)])


def new_baseline(wave_ch=None):
    """파형 채널이 있으면 WaveBaseline, 없으면 기존 SegmentBaseline"""
    return WaveBaseline(wave_ch) if wave_ch else SegmentBaseline()


# ------------------------------------------------------------ 기존 모델 연결 (앞쪽 특징 열만 넘김)
class SlicedDetector:
    """기존 특징 기반 이상 감지 모델에 Z의 앞쪽 nf열(특징 z)만 넘긴다. 눈금 값(r0·r995·rfail)은 이 객체에 붙는다."""

    def __init__(self, det, nf):
        self.det, self.nf = det, nf
        self.name = getattr(det, "name", "")

    def fit(self, Z):
        self.det.fit(Z[:, :self.nf])
        return self

    def raw(self, Z):
        return self.det.raw(Z[:, :self.nf])


class SlicedCause:
    """랜덤포레스트 원인 분류에 특징 열만 넘긴다 (CAUSE_MODEL = 'rf'일 때)"""

    def __init__(self, cause, nf):
        self.cause, self.nf = cause, nf
        self.classes = cause.classes

    def proba(self, Z):
        return self.cause.proba(np.atleast_2d(Z)[:, :self.nf])

    def importance(self):
        return self.cause.importance()

    def explain(self, z, label, top=3):
        return self.cause.explain(np.asarray(z)[:self.nf], label, top)


# ------------------------------------------------------------ PyTorch 신경망
def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("딥러닝 모델(config.DEEP_MODEL = True)에는 PyTorch가 필요합니다.\n"
                          "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                          "PyTorch 없이 돌리려면 config.DEEP_MODEL = False로 바꾸세요 (기존 특징 + 오토인코더 + 랜덤포레스트).") from e
    if not getattr(_torch, "_seeded", False):
        torch.manual_seed(C.SEED)
        _torch._seeded = True
    return torch


def _build(kind, cin, L, nf=0, n_out=0):
    """kind = 'ae' 또는 'clf'. 인코더: 합성곱 4층 (stride 2씩 → 길이 L/16), 첫 층은 넓은 커널로 진동 주기를 잡음"""
    torch = _torch()
    nn = torch.nn

    def block(i, o, k):
        return [nn.Conv1d(i, o, k, stride=2, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]

    enc = nn.Sequential(*block(cin, 16, 15), *block(16, 32, 9), *block(32, 64, 5), *block(64, 64, 5))
    Ls = L // 16

    if kind == "ae":
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = enc
                self.to_z = nn.Linear(64 * Ls, C.DEEP_LATENT)
                self.from_z = nn.Linear(C.DEEP_LATENT, 64 * Ls)

                def up(i, o, last=False):
                    layers = [nn.ConvTranspose1d(i, o, 4, stride=2, padding=1)]
                    return layers if last else layers + [nn.BatchNorm1d(o), nn.ReLU()]
                self.dec = nn.Sequential(*up(64, 64), *up(64, 32), *up(32, 16), *up(16, cin, last=True))

            def forward(self, x):
                h = self.enc(x)
                z = self.to_z(h.flatten(1))
                h = self.from_z(z).view(-1, 64, Ls)
                return self.dec(h)
        return AE()

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.fnet = nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Dropout(0.2)) if nf else None
            self.head = nn.Sequential(nn.Linear(128 + (64 if nf else 0), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_out))

        def forward(self, x, f=None):
            h = self.enc(x)
            h = torch.cat([h.mean(-1), h.amax(-1)], 1)      # 시간 방향 평균 + 최대 (어느 시점에 나와도 잡음)
            if self.fnet is not None:
                h = torch.cat([h, self.fnet(f)], 1)
            return self.head(h)
    return Clf()


def _device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchModel:
    """PyTorch 모델을 joblib로 저장할 수 있게: 저장할 때는 가중치(state_dict)만 numpy로, 불러올 때 다시 만든다"""
    _net = None

    def _make(self):
        raise NotImplementedError

    def net(self):
        if self._net is None:
            torch = _torch()
            self._net = self._make().to(_device(torch))
            if getattr(self, "_state", None) is not None:
                self._net.load_state_dict({k: torch.as_tensor(v) for k, v in self._state.items()})
            self._net.eval()
        return self._net

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k != "_net"}
        if self._net is not None:
            d["_state"] = {k: v.detach().cpu().numpy() for k, v in self._net.state_dict().items()}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._net = None


def _split(n, rng, val=0.15):
    idx = rng.permutation(n)
    k = max(1, int(n * val)) if n >= 20 else 0
    return idx[k:], idx[:k]


def _train(net, loss_fn, n, batch_fn):
    """공통 학습 루프: Adam, 검증 손실이 DEEP_PATIENCE번 연속 안 줄면 멈추고 가장 좋던 가중치로 되돌림"""
    torch = _torch()
    rng = np.random.default_rng(C.SEED)
    tr, va = _split(n, rng)
    opt = torch.optim.Adam(net.parameters(), lr=C.DEEP_LR, weight_decay=1e-4)
    best, best_state, bad = None, None, 0
    for ep in range(C.DEEP_EPOCHS):
        net.train()
        for i in range(0, len(tr), C.DEEP_BATCH):
            b = tr[i:i + C.DEEP_BATCH]
            if len(b) < 2:                                  # BatchNorm은 2개 이상 필요
                continue
            opt.zero_grad()
            loss = loss_fn(*batch_fn(b, train=True))
            loss.backward()
            opt.step()
        rng.shuffle(tr)
        if not len(va):
            continue
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_fn(*batch_fn(va[i:i + 512], train=False))) for i in range(0, len(va), 512)]))
        if best is None or vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= C.DEEP_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return ep + 1, best


def _limit(n, rng):
    """학습 창이 너무 많으면 무작위로 DEEP_MAX_WINDOWS개만 (메모리·시간)"""
    return np.sort(rng.choice(n, C.DEEP_MAX_WINDOWS, replace=False)) if n > C.DEEP_MAX_WINDOWS else np.arange(n)


class CNNAEDetector(_TorchModel):
    """1D-CNN 오토인코더: 정상 파형만으로 '복원'을 배우고, 잘 복원되지 않는 정도(평균 제곱 오차)를 이상 점수로 쓴다."""
    name = "1D-CNN 오토인코더 (파형)"

    def __init__(self, nf, wave_ch):
        self.nf, self.wave_ch, self.L = nf, list(wave_ch), C.DEEP_WAVE_LEN

    def _make(self):
        return _build("ae", len(self.wave_ch), self.L)

    def _W(self, Z):
        return np.asarray(Z, np.float32)[:, self.nf:].reshape(len(Z), len(self.wave_ch), self.L)

    def fit(self, Z):
        torch = _torch()
        dev = _device(torch)
        W = self._W(Z)[_limit(len(Z), np.random.default_rng(C.SEED))]
        self._net = self._make().to(dev)
        mse = torch.nn.MSELoss()

        def batch(b, train):
            x = torch.as_tensor(W[b], device=dev)
            return self._net(x), x
        self.epochs, self.val_loss = _train(self._net, mse, len(W), batch)
        return self

    def raw(self, Z):
        torch = _torch()
        net, dev = self.net(), _device(torch)
        W, out = self._W(np.atleast_2d(Z)), []
        with torch.no_grad():
            for i in range(0, len(W), 512):
                x = torch.as_tensor(W[i:i + 512], device=dev)
                out.append(((net(x) - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)


class MaxDetector:
    """두 모델의 건강 점수 중 큰 값 (특징 오토인코더가 놓친 것을 파형 모델이, 그 반대도 잡도록). 합친 점수로 눈금을 다시 맞춤"""
    name = "특징 AI + 1D-CNN (주 모델)"

    def __init__(self, a, b):
        self.a, self.b = a, b

    def fit(self, Z):
        return self

    def raw(self, Z):
        from pipeline.models import health
        return np.maximum(health(self.a, Z), health(self.b, Z))


class DeepCauseClassifier(_TorchModel):
    """1D-CNN(파형) + 작은 신경망(특징 z) → 원인 확률. 파형 쪽 인코더 구조는 오토인코더와 같지만 가중치는 따로 학습한다
    (같이 쓰면 원인 학습이 감지 모델을 고장 데이터 쪽으로 끌고 가 '처음 보는 고장' 감지가 약해짐)."""
    use_last = True                                         # 판단 근거는 주기 평균이 아니라 마지막 창으로 (파형은 평균 내면 지워짐)

    def __init__(self, nf, wave_ch):
        self.nf, self.wave_ch, self.L = nf, list(wave_ch), C.DEEP_WAVE_LEN

    def _make(self):
        return _build("clf", len(self.wave_ch), self.L, nf=self.nf if C.DEEP_CAUSE_FEATS else 0, n_out=len(self.classes))

    def _split_in(self, Z):
        Z = np.asarray(np.atleast_2d(Z), np.float32)
        return Z[:, self.nf:].reshape(len(Z), len(self.wave_ch), self.L), Z[:, :self.nf]

    def fit(self, Z, labels, feats=None):
        torch = _torch()
        dev = _device(torch)
        rng = np.random.default_rng(C.SEED)
        keep = _limit(len(Z), rng)
        W, F = self._split_in(np.asarray(Z)[keep])
        labels = np.asarray(labels)[keep]
        self.feats = list(feats or [])
        self.classes = sorted({str(c) for c in labels})
        y = np.array([self.classes.index(str(c)) for c in labels])
        cnt = np.bincount(y, minlength=len(self.classes)).astype(float)
        wts = torch.as_tensor(len(y) / (len(self.classes) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)   # 클래스 균형
        self._net = self._make().to(dev)
        ce = torch.nn.CrossEntropyLoss(weight=wts)
        use_f = C.DEEP_CAUSE_FEATS

        def batch(b, train):
            x = W[b]
            if train:                                       # 데이터 늘리기: 시간 이동 + 크기 ±10%
                x = np.roll(x, int(rng.integers(self.L)), axis=2) * rng.uniform(0.9, 1.1, (len(b), 1, 1)).astype(np.float32)
            x = torch.as_tensor(np.ascontiguousarray(x), device=dev)
            f = torch.as_tensor(F[b], device=dev) if use_f else None
            return self._net(x, f), torch.as_tensor(y[b], dtype=torch.long, device=dev)
        self.epochs, self.val_loss = _train(self._net, ce, len(y), batch)
        self._importance = self._attr_summary(np.asarray(Z)[keep][rng.permutation(len(keep))[:400]])
        return self

    def _forward(self, Z):
        torch = _torch()
        dev = _device(torch)
        W, F = self._split_in(Z)
        x = torch.as_tensor(W, device=dev)
        f = torch.as_tensor(F, device=dev) if C.DEEP_CAUSE_FEATS else None
        return self.net()(x, f)

    def proba(self, Z):
        torch = _torch()
        Z, out = np.atleast_2d(Z), []
        with torch.no_grad():
            for i in range(0, len(Z), 512):
                out.append(torch.softmax(self._forward(Z[i:i + 512]), 1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.classes)))

    def _attr(self, Z, cls_idx=None):
        """입력 × 기울기: 그 원인 확률을 올리는 데 입력의 어느 부분이 기여했나 → (파형 [N, 채널, L], 특징 [N, nf])"""
        torch = _torch()
        dev = _device(torch)
        net = self.net()
        W, F = self._split_in(Z)
        x = torch.as_tensor(W, device=dev).requires_grad_(True)
        f = torch.as_tensor(F, device=dev).requires_grad_(True) if C.DEEP_CAUSE_FEATS else None
        logits = net(x, f)
        idx = logits.argmax(1) if cls_idx is None else torch.full((len(W),), cls_idx, device=dev, dtype=torch.long)
        logits.gather(1, idx[:, None]).sum().backward()
        aw = (x.grad * x).detach().cpu().numpy()
        af = (f.grad * f).detach().cpu().numpy() if f is not None else np.zeros((len(W), 0))
        return aw, af

    def _attr_summary(self, Z):
        aw, af = self._attr(Z)
        items = [(wave_ko(ch), float(np.mean(np.abs(aw[:, i])))) for i, ch in enumerate(self.wave_ch)]
        items += [(f, float(np.mean(np.abs(af[:, j])))) for j, f in enumerate(self.feats[:af.shape[1]])]
        tot = sum(v for _, v in items) or 1.0
        return sorted(((n, v / tot) for n, v in items), key=lambda x: -x[1])

    def importance(self):
        """전체적으로 판정에 많이 쓰인 입력 순위 (fig_importance용, 합 = 1)"""
        return list(getattr(self, "_importance", []))

    def explain(self, z, label, top=3):
        """이 판정(label)을 밀어 올린 입력 top개: '파형 채널 + 0.2초 구간' 또는 특징 이름 → [(이름, 기여도)]"""
        aw, af = self._attr(np.atleast_2d(z), self.classes.index(label))
        seg = self.L // 5
        items = [(f"{wave_ko(ch)} {s * 0.2:.1f}~{(s + 1) * 0.2:.1f}초", float(aw[0, i, s * seg:(s + 1) * seg].sum()))
                 for i, ch in enumerate(self.wave_ch) for s in range(5)]
        items += [(f, float(af[0, j])) for j, f in enumerate(self.feats[:af.shape[1]])]
        return sorted(items, key=lambda x: -x[1])[:top]
