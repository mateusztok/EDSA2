import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
import warnings

# Ignorowanie ostrzeżeń związanych z dzieleniem przez zero w logarytmach
warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURACJA I PARAMETRY
# ==========================================
N = 10000  # Minimalna liczba próbek
DT = 0.01  # Krok czasowy
LAG_MAX = 100


# ==========================================
# 2. GENERATORY SYGNAŁÓW
# ==========================================

def gen_random(n):
    """Sygnał losowy o rozkładzie jednostajnym."""
    return np.random.uniform(0, 1, n)


def gen_periodic(n):
    """Sygnał periodyczny."""
    t = np.linspace(0, 100, n)
    return np.sin(t) + np.cos(t)


def gen_lorenz(n, dt):
    """System chaotyczny Lorenza."""

    def lorenz_ode(state, t):
        x, y, z = state
        return [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z]

    sol = odeint(lorenz_ode, [1.0, 1.0, 1.0], np.arange(0, n * dt, dt))
    return sol[:, 0]


def gen_lu(n, dt):
    """System chaotyczny Lü (klasyczny)."""

    def lu_ode(state, t, a=36, b=3, c=20):
        x, y, z = state
        return [a * (y - x), -x * z + c * y, x * y - b * z]

    sol = odeint(lu_ode, [0.2, 0.5, 0.3], np.arange(0, n * dt, dt))
    return sol[:, 0]


def gen_lu_fractional(q, n, dt):
    """Ułamkowy system Lü (metoda Grunwalda-Letnikova)."""
    a, b, c = 36, 3, 20
    x = np.zeros((n, 3))
    x[0] = [0.2, 0.5, 0.3]
    cp = [1.0]
    for j in range(1, n):
        cp.append(cp[-1] * (1 - (q + 1) / j))

    for i in range(1, n):
        # Implementacja pamięci systemu dla pochodnej ułamkowej
        mem = -np.dot(cp[1:i + 1], x[i - 1::-1, :])
        d = np.array([a * (x[i - 1, 1] - x[i - 1, 0]), -x[i - 1, 0] * x[i - 1, 2] + c * x[i - 1, 1],
                      x[i - 1, 0] * x[i - 1, 1] - b * x[i - 1, 2]])
        x[i] = d * (dt ** q) + mem
    return x[:, 0]


# ==========================================
# 3. METODY ANALITYCZNE
# ==========================================

def get_mi_analysis(sig, max_lag=LAG_MAX):
    """Estymacja opóźnienia T przy użyciu Wzajemnej Informacji."""
    mi = []
    for lag in range(1, max_lag + 1):
        hist_2d, _, _ = np.histogram2d(sig[:-lag], sig[lag:], bins=20)
        p_xy = hist_2d / (np.sum(hist_2d) + 1e-12)
        p_x, p_y = np.sum(p_xy, axis=1), np.sum(p_xy, axis=0)
        idx = p_xy > 0
        mi.append(np.sum(p_xy[idx] * np.log(p_xy[idx] / (np.outer(p_x, p_y)[idx] + 1e-12))))
    T_opt = np.argmin(mi) + 1
    return T_opt, mi


def calculate_hurst(ts):
    """Analiza trendów wykładnika Hursta."""
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
        if chunk_rs: rs_res.append(np.mean(chunk_rs))
    return np.polyfit(np.log(lags[:len(rs_res)]), np.log(rs_res), 1)[0]


def analyze_fractals(sig, T, de):
    """Całka korelacyjna, wymiar D2 i entropia K2."""
    rs = np.logspace(-1.5, 0.5, 10)
    cr_results = {}
    for d in [de, de + 1]:
        m_len = len(sig) - (d - 1) * T
        vectors = np.array([sig[i:i + (d - 1) * T + 1:T] for i in range(m_len)])
        idx = np.random.choice(m_len, min(m_len, 500), replace=False)
        dists = pdist(vectors[idx])
        cr_results[d] = np.array([np.sum(dists < r) / len(dists) for r in rs])

    # D2 jako nachylenie w skali log-log
    valid = cr_results[de] > 0
    d2 = np.polyfit(np.log(rs[valid]), np.log(cr_results[de][valid]), 1)[0]

    # Entropia korelacyjna K2
    mid = len(rs) // 2
    k2 = (1 / (T * DT)) * np.log(cr_results[de][mid] / cr_results[de + 1][mid]) if cr_results[de + 1][mid] > 0 else 0
    return d2, k2, rs, cr_results[de]


def calculate_box_dimension(sig, T, de):
    """Wyznaczanie wymiaru pojemnościowego (D0) metodą pudełkową (box-counting)."""
    m_len = len(sig) - (de - 1) * T
    vectors = np.array([sig[i:i + (de - 1) * T + 1:T] for i in range(m_len)])

    # Normalizacja danych do zakresu [0, 1] w celu ujednolicenia siatki pudełek
    v_min, v_max = vectors.min(axis=0), vectors.max(axis=0)
    vectors_norm = (vectors - v_min) / (v_max - v_min + 1e-12)

    # Definicja różnych wielkości pudełek (skal epsilon)
    epsilons = np.logspace(-2, -0.5, 10)
    counts = []

    for eps in epsilons:
        # Wyznaczanie współrzędnych pudełek (indeksowanie siatki)
        # Każdy punkt wpada do pudełka o indeksach floor(coord / eps)
        boxes = np.floor(vectors_norm / eps).astype(int)
        # Liczenie unikalnych (zajętych) pudełek w przestrzeni de-wymiarowej
        unique_boxes = np.unique(boxes, axis=0)
        counts.append(len(unique_boxes))

    # D0 to nachylenie prostej: log(N) = D0 * log(1/eps) => log(N) = -D0 * log(eps)
    # Wykonujemy dopasowanie liniowe w skali log-log
    coeffs = np.polyfit(np.log(epsilons), np.log(counts), 1)
    return -coeffs[0]


def calculate_lle(sig, T, de, dt, n_steps=10):
    """Estymacja największego wykładnika Lapunowa (metoda Rosensteina)."""
    m_len = len(sig) - (de - 1) * T
    vectors = np.array([sig[i:i + (de - 1) * T + 1:T] for i in range(m_len)])
    max_idx = m_len - n_steps - 1
    if max_idx <= 0: return 0

    n_pts = min(max_idx, 400)
    indices = np.random.choice(max_idx, n_pts, replace=False)
    theiler = T * (de - 1)
    d_evolution = np.zeros(n_steps)
    counts = np.zeros(n_steps)

    for i in indices:
        dists = np.linalg.norm(vectors[:max_idx] - vectors[i], axis=1)
        dists[max(0, i - theiler): min(max_idx, i + theiler)] = np.inf
        neighbor_idx = np.argmin(dists)
        if dists[neighbor_idx] < np.inf and dists[neighbor_idx] > 0:
            for s in range(n_steps):
                d = np.linalg.norm(vectors[i + s] - vectors[neighbor_idx + s])
                if d > 0:
                    d_evolution[s] += np.log(d)
                    counts[s] += 1
    valid = counts > 0
    if not any(valid): return 0
    lle = np.polyfit(np.arange(np.sum(valid)) * dt, d_evolution[valid] / counts[valid], 1)[0]
    return lle


# ==========================================
# 4. WIZUALIZACJA I GŁÓWNA PĘTLA
# ==========================================

def run_full_analysis():
    signals = {
        "Losowy": gen_random(N),
        "Periodyczny": gen_periodic(N),
        "Lorenz": gen_lorenz(N, DT),
        "Lu_Klasyczny": gen_lu(N, DT),
        "Lu_Ulamkowy_q95": gen_lu_fractional(0.95, N, DT),
        "Lu_Ulamkowy_q85": gen_lu_fractional(0.85, N, DT)
    }

    final_results = []

    for name, sig in signals.items():
        print(f"Analiza: {name}...")

        T, mi_data = get_mi_analysis(sig)
        de = 2 if name == "Periodyczny" else 3
        h_val = calculate_hurst(sig)
        d2, k2, rs, cr_data = analyze_fractals(sig, T, de)
        d0_val = calculate_box_dimension(sig, T, de)  # Nowy parametr: D0
        lle = calculate_lle(sig, T, de, DT)

        final_results.append({
            "Sygnał": name, "T": T, "dE": de,
            "Hurst": round(h_val, 4), "D0": round(d0_val, 4),
            "D2": round(d2, 4), "K2": round(k2, 4), "LLE": round(lle, 4)
        })

        # Wykresy diagnostyczne
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"Raport: {name}", fontsize=14)

        # 1. MI Plot
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(mi_data);
        ax1.axvline(T, color='r', ls='--');
        ax1.set_title("Wzajemna Informacja")

        # 2. Log-Log Correlation
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.loglog(rs, cr_data, 'o-');
        ax2.set_title("Całka korelacyjna (log-log)")

        # 3. Atraktor 3D
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot(sig[:-2 * T], sig[T:-T], sig[2 * T:], lw=0.5)
        ax3.set_title("Rekonstrukcja 3D")

        # 4. Series fragment
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(sig[:1000]);
        ax4.set_title("Fragment szeregu")

        plt.tight_layout()
        plt.savefig(f"wykres_{name}.png")
        plt.close()

    # Tabela zbiorcza
    df = pd.DataFrame(final_results)
    print("\n--- ZBIORCZE WYNIKI ---")
    print(df)
    df.to_csv('wyniki_projekt_2.csv', index=False)

if __name__ == "__main__":
    run_full_analysis()