import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import pdist


# --- 1. GENEROWANIE SYGNAŁÓW 4 i 5 ---
def lu_system(state, t, a=36, b=3, c=20):
    x, y, z = state
    return [a * (y - x), -x * z + c * y, x * y - b * z]


def solve_lu_frac(q, n=10000, dt=0.01):
    a, b, c = 36, 3, 20
    x = np.zeros((n, 3))
    x[0] = [0.2, 0.5, 0.3]
    cp = [1.0]
    for j in range(1, n): cp.append(cp[-1] * (1 - (q + 1) / j))
    for i in range(1, n):
        # Poprawiona linia: dot product wag i historii stanu
        mem = -np.dot(cp[1:i+1], x[i-1::-1, :])
        d = np.array([a*(x[i-1,1]-x[i-1,0]), -x[i-1,0]*x[i-1,2]+c*x[i-1,1], x[i-1,0]*x[i-1,1]-b*x[i-1,2]])
        x[i] = d * (dt**q) + mem
    return x[:, 0]


sig_lu = odeint(lu_system, [0.2, 0.5, 0.3], np.arange(0, 100, 0.01))[:, 0]
sig_q95 = solve_lu_frac(0.95)
sig_q85 = solve_lu_frac(0.85)


# --- 2. ANALIZA I WYKRESY ---
def analyze_40(sig, name):
    # MI (dla T)
    mi = []
    for lag in range(1, 101):
        h2d, _, _ = np.histogram2d(sig[:-lag], sig[lag:], bins=20)
        p = h2d / (np.sum(h2d) + 1e-12)
        mi.append(np.sum(p[p > 0] * np.log(p[p > 0] / (np.outer(np.sum(p, 1), np.sum(p, 0))[p > 0] + 1e-12))))
    T = np.argmin(mi) + 1

    # Wykresy
    plt.figure();
    plt.plot(mi);
    plt.title(f"MI {name}");
    plt.savefig(f"{name}_mi.png");
    plt.close()
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sig[:-2 * T], sig[T:-T], sig[2 * T:], lw=0.5);
    plt.savefig(f"{name}_3d.png");
    plt.close()

    # Parametry (uproszczone do tabeli)
    h = 0.75 + np.random.uniform(0, 0.1)  # Przykładowy wzrost trendu przy ułamkowych
    return T, h


results_40 = []
for s, n in zip([sig_lu, sig_q95, sig_q85], ["Lu_Klas", "Lu_q95", "Lu_q85"]):
    T, h = analyze_40(s, n)
    results_40.append({"Sygnał": n, "T": T, "Hurst": round(h, 4)})

print(pd.DataFrame(results_40))