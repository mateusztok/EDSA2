import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- KONFIGURACJA ---
N = 10000  # Minimalna liczba próbek [cite: 60]
dt = 0.01  # Krok czasowy

# 1. SYGNAŁ LOSOWY (Rozkład jednostajny 0-1) [cite: 11]
sig_random = np.random.uniform(0, 1, N)

# 2. SYGNAŁ PERIODYCZNY (sin(n) + cos(n)) [cite: 12]
n_axis = np.linspace(0, 100, N)
sig_periodic = np.sin(n_axis) + np.cos(n_axis)

# 3. SYSTEM LORENZA (Klasyczny) [cite: 13]
def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

t_span = np.arange(0, N * dt, dt)
sol_lorenz = odeint(lorenz, [1.0, 1.0, 1.0], t_span)
sig_lorenz = sol_lorenz[:, 0]  # Szereg czasowy zmiennej x

# 4. SYSTEM LÜ (Klasyczny - Twój wybór) [cite: 101]
def lu_system(state, t, a=36, b=3, c=20):
    x, y, z = state
    return [a * (y - x), -x * z + c * y, x * y - b * z]

# Warunki początkowe z przygotowania: [0.2, 0.5, 0.3] [cite: 101]
sol_lu = odeint(lu_system, [0.2, 0.5, 0.3], t_span)
sig_lu = sol_lu[:, 0]  # Szereg czasowy zmiennej x

# 5. UŁAMKOWY SYSTEM LÜ (Metoda Grunwalda-Letnikova) [cite: 15, 103]
def solve_fractional_lu(q, n_samples, dt, x0):
    a, b, c = 36, 3, 20
    x = np.zeros((n_samples, 3))
    x[0] = x0
    # Współczynniki dwumianowe dla GL
    cp = [1.0]
    for j in range(1, n_samples):
        cp.append(cp[-1] * (1 - (q + 1) / j))
    for i in range(1, n_samples):
        # Pamięć systemu (istota pochodnej ułamkowej)
        mem_x = -np.dot(cp[1:i+1], x[i-1::-1, 0])
        mem_y = -np.dot(cp[1:i+1], x[i-1::-1, 1])
        mem_z = -np.dot(cp[1:i+1], x[i-1::-1, 2])
        x[i, 0] = (a * (x[i-1, 1] - x[i-1, 0])) * (dt**q) + mem_x
        x[i, 1] = (-x[i-1, 0] * x[i-1, 2] + c * x[i-1, 1]) * (dt**q) + mem_y
        x[i, 2] = (x[i-1, 0] * x[i-1, 1] - b * x[i-1, 2]) * (dt**q) + mem_z
    return x

# Generowanie dwóch potęg ułamkowych z przygotowania [cite: 103]
sig_frac_95 = solve_fractional_lu(0.95, N, 0.01, [0.2, 0.5, 0.3])[:, 0]
sig_frac_85 = solve_fractional_lu(0.85, N, 0.01, [0.2, 0.5, 0.3])[:, 0]

# --- RYSOWANIE (Wizualizacja do sprawozdania) ---
fig, axes = plt.subplots(6, 1, figsize=(10, 15))
signals = [sig_random, sig_periodic, sig_lorenz, sig_lu, sig_frac_95, sig_frac_85]
titles = ["Losowy", "Periodyczny", "Lorenz (x)", "Lü (x)", "Lü Ułamkowy q=0.95 (x)", "Lü Ułamkowy q=0.85 (x)"]

for i in range(6):
    axes[i].plot(signals[i][:2000], lw=0.8) # Rysujemy fragment dla przejrzystości
    axes[i].set_title(titles[i])
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Opcjonalnie: Atraktor 3D dla systemu Lü (wymagane w raporcie) [cite: 46]
fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(sol_lu[:, 0], sol_lu[:, 1], sol_lu[:, 2], color='red', lw=0.5)
ax.set_title("Oryginalny Atraktor Lü (Klasyczny)")
plt.show()