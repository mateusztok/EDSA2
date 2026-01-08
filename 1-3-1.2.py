import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. GENEROWANIE SYGNAŁÓW (N = 10 000) [cite: 60]
# ==========================================
N, dt = 10000, 0.01

# Sygnał 1: Losowy [cite: 11]
sig_random = np.random.uniform(0, 1, N)

# Sygnał 2: Periodyczny [cite: 12]
t_p = np.linspace(0, 100, N)
sig_periodic = np.sin(t_p) + np.cos(t_p)


# Sygnał 3: Lorenz [cite: 13]
def lorenz(state, t):
    x, y, z = state
    return [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z]


sig_lorenz = odeint(lorenz, [1.0, 1.0, 1.0], np.arange(0, N * dt, dt))[:, 0]

signals = {"Losowy": sig_random, "Periodyczny": sig_periodic, "Lorenz": sig_lorenz}


# ==========================================
# 2. FUNKCJE ANALITYCZNE (PUNKT 1.2) [cite: 18-27]
# ==========================================

def get_mi_analysis(x, name):
    """Estymacja opóźnienia T (Mutual Information)[cite: 21]."""
    max_lag = 100
    mi = []
    for lag in range(1, max_lag + 1):
        hist_2d, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=20)
        p_xy = hist_2d / (np.sum(hist_2d) + 1e-12)
        p_x, p_y = np.sum(p_xy, axis=1), np.sum(p_xy, axis=0)
        idx = p_xy > 0
        mi.append(np.sum(p_xy[idx] * np.log(p_xy[idx] / (np.outer(p_x, p_y)[idx] + 1e-12))))

    T_opt = np.argmin(mi) + 1

    plt.figure()
    plt.plot(range(1, max_lag + 1), mi)
    plt.axvline(T_opt, color='r', linestyle='--', label=f'T={T_opt}')
    plt.title(f"Wzajemna Informacja - {name}")
    plt.xlabel("Lag");
    plt.ylabel("MI");
    plt.legend()
    plt.savefig(f"{name}_mi.png")
    plt.close()
    return T_opt


def calculate_hurst_fixed(ts):
    """Poprawiona funkcja Hursta[cite: 22]."""
    n = len(ts)
    lags = np.unique(np.logspace(1, np.log10(n / 3), 10).astype(int))
    rs_res = []
    for l in lags:
        chunks = [ts[i:i + l] for i in range(0, n - l, l)]
        chunk_rs = []
        for c in chunks:
            s = np.std(c)
            if s > 0:
                y = np.cumsum(c - np.mean(c))
                r = np.max(y) - np.min(y)
                chunk_rs.append(r / s)
        if chunk_rs:
            rs_res.append(np.mean(chunk_rs))
    return np.polyfit(np.log(lags[:len(rs_res)]), np.log(rs_res), 1)[0]


def analyze_full(sig, T, de, name):
    """Analiza całki korelacyjnej, atraktora i fraktali [cite: 19, 20, 24-27]."""
    # Wykres całki korelacyjnej
    rs = np.logspace(-1.5, 0.5, 10)
    plt.figure()
    for d in [de, de + 1]:
        M = len(sig) - (d - 1) * T
        vectors = np.array([sig[i:i + (d - 1) * T + 1:T] for i in range(M)])
        idx = np.random.choice(M, min(M, 500), replace=False)
        dists = pdist(vectors[idx])
        cr = np.array([np.sum(dists < r) / len(dists) for r in rs])
        plt.loglog(rs, cr, '-o', label=f'd={d}')
        if d == de: cr_main = cr
        if d == de + 1: cr_plus = cr
    plt.title(f"Całka korelacyjna (skala log-log) - {name}")
    plt.xlabel("r");
    plt.ylabel("C(r)");
    plt.legend();
    plt.grid(True)
    plt.savefig(f"{name}_ci.png");
    plt.close()

    # Atraktor 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sig[:-2 * T], sig[T:-T], sig[2 * T:], lw=0.5)
    ax.set_title(f"Zrekonstruowana przestrzeń fazowa - {name}")
    plt.savefig(f"{name}_3d.png");
    plt.close()

    # Obliczenia parametrów
    h = calculate_hurst_fixed(sig)
    d2 = np.polyfit(np.log(rs[cr_main > 0]), np.log(cr_main[cr_main > 0]), 1)[0]
    k2 = (1 / T) * np.log(cr_main[len(rs) // 2] / cr_plus[len(rs) // 2]) if cr_plus[len(rs) // 2] > 0 else 0

    return h, d2, k2


# ==========================================
# 3. URUCHOMIENIE ANALIZY
# ==========================================
final_results = []

for name, sig in signals.items():
    print(f"Analiza sygnału: {name}...")
    T = get_mi_analysis(sig, name)
    # Wymiar zanurzenia de: 3 dla chaosu/losowości, 2 dla periodyczności [cite: 20]
    de = 3 if name != "Periodyczny" else 2

    h, d2, k2 = analyze_full(sig, T, de, name)

    final_results.append({
        "Sygnał": name, "T": T, "dE": de,
        "Hurst": round(h, 4), "D2": round(d2, 4), "K2": round(k2, 4)
    })

df = pd.DataFrame(final_results)
print("\n--- ZBIORCZE WYNIKI DLA SYGNAŁÓW 1, 2, 3 ---")
print(df)
df.to_csv('analiza_123_wyniki.csv', index=False)