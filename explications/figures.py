"""Génère les figures et vérifications numériques du document d'explications (Chapitre 2, MCO)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from math import erf

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Serif"})
out = "figures/"

# ---------- 1. Exemple numérique (n=5) ----------
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 4, 5, 4, 5.])
X = np.column_stack([np.ones(5), x])
b = np.linalg.solve(X.T @ X, X.T @ y)
e = y - X @ b
print("b =", b, " e =", e, " X'e =", X.T @ e, " SCE =", e @ e)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, color="C0", zorder=3, label="Observations $(x_i, y_i)$")
xx = np.linspace(0.5, 5.5, 50)
ax.plot(xx, b[0] + b[1] * xx, color="C3", label=r"Droite MCO $\hat y = 2{,}2 + 0{,}6x$")
for xi, yi, fi in zip(x, y, X @ b):
    ax.plot([xi, xi], [yi, fi], "k--", lw=1)
ax.scatter([3], [4], marker="*", s=200, color="C2", zorder=4, label=r"Point moyen $(\bar x, \bar y)=(3,4)$")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Résidus $e_i$ (pointillés) : les MCO minimisent $\\sum e_i^2$")
fig.tight_layout(); fig.savefig(out + "exemple_droite.pdf"); plt.close(fig)

# ---------- 2. Géométrie de la projection ----------
fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(111, projection="3d")
Xp, Yp = np.meshgrid(np.linspace(-0.5, 3, 2), np.linspace(-0.5, 3, 2))
ax.plot_surface(Xp, Yp, 0 * Xp, alpha=0.15, color="C0")
yv = np.array([2.2, 1.6, 2.4]); yhat = np.array([2.2, 1.6, 0])
o = np.zeros(3)
def arrow(a, bb, c, lab, off=(0, 0, 0)):
    ax.quiver(*a, *(bb - a), color=c, arrow_length_ratio=0.08, lw=2)
    ax.text(*(bb + np.array(off)), lab, color=c, fontsize=13)
arrow(o, yv, "k", "$y$", (0.05, 0, 0.1))
arrow(o, yhat, "C3", r"$\hat y = Py = Xb$", (0.05, 0.05, -0.35))
arrow(yhat, yv, "C2", "$e = My$", (0.05, 0, -1.2))
arrow(o, np.array([2.8, 0, 0]), "C0", "$x_1$")
arrow(o, np.array([0, 2.8, 0]), "C0", "$x_2$")
ax.plot([2.2, 2.2], [1.6, 1.6], [0, 0.25], "k")
ax.text(1.0, 2.6, 0, "Espace engendré\npar les colonnes de $X$", color="C0", fontsize=9)
ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_zlim(0, 2.6)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=18, azim=-60)
ax.set_title("$y = Py + My$ : projection orthogonale de $y$ sur Col$(X)$")
fig.tight_layout(); fig.savefig(out + "projection.pdf"); plt.close(fig)

# ---------- 3. R² et variables de bruit ----------
rng = np.random.default_rng(2026)
n = 50
x1 = rng.normal(size=n); yb = 1 + 0.5 * x1 + rng.normal(size=n)
Xk = np.column_stack([np.ones(n), x1])
r2, r2a, ks = [], [], []
sct = ((yb - yb.mean()) ** 2).sum()
for j in range(0, 46):
    if j > 0:
        Xk = np.column_stack([Xk, rng.normal(size=n)])
    bk = np.linalg.lstsq(Xk, yb, rcond=None)[0]
    sce = ((yb - Xk @ bk) ** 2).sum()
    K = Xk.shape[1]
    r2.append(1 - sce / sct); r2a.append(1 - (sce / (n - K)) / (sct / (n - 1))); ks.append(j)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(ks, r2, "C3-o", ms=3, label="$R^2$")
ax.plot(ks, r2a, "C0-s", ms=3, label=r"$\bar R^2$ (ajusté)")
ax.axhline(r2[0], color="gray", ls=":")
ax.set_xlabel("Nombre de variables de bruit pur ajoutées ($n = 50$)")
ax.set_ylabel("Valeur"); ax.set_ylim(-0.2, 1.02); ax.legend()
ax.set_title("Le $R^2$ ne baisse jamais, même avec des variables inutiles")
fig.tight_layout(); fig.savefig(out + "r2_bruit.pdf"); plt.close(fig)
print("R2 sans bruit", r2[0], "avec 45 bruits", r2[-1], "R2adj", r2a[0], r2a[-1])

# ---------- 4. FWL vérification ----------
n = 200
x1 = rng.normal(size=n); x2 = 0.7 * x1 + rng.normal(size=n)
yf = 1 + 2 * x1 - 1 * x2 + rng.normal(size=n)
Xf = np.column_stack([np.ones(n), x1, x2])
bf = np.linalg.lstsq(Xf, yf, rcond=None)[0]
X1 = np.column_stack([np.ones(n), x1])
M1 = np.eye(n) - X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
b2_fwl = (M1 @ x2) @ (M1 @ yf) / ((M1 @ x2) @ (M1 @ x2))
print("FWL: b2 complet =", bf[2], " b2 FWL =", b2_fwl)

# ---------- 5. Monte Carlo (réplique du cours) ----------
rng = np.random.default_rng(12345678)
beta0 = beta1 = sigma = 1.0; R = 10000; N = 250
xm = np.exp(rng.normal(0, 1, N))
Sxx = ((xm - xm.mean()) ** 2).sum()
sd_true = np.sqrt(sigma ** 2 / Sxx)
Xm = np.column_stack([np.ones(N), xm]); XtXi = np.linalg.inv(Xm.T @ Xm)
eps = rng.normal(0, sigma, (R, N))
Y = beta0 + beta1 * xm + eps
B = Y @ Xm @ XtXi            # R x 2 (XtXi symétrique)
res = Y - B @ Xm.T
s2 = (res ** 2).sum(1) / (N - 2)
se = np.sqrt(s2 * XtXi[1, 1])
b1s = B[:, 1]
print(f"MC: moyenne b1={b1s.mean():.6f} ET b1={b1s.std(ddof=1):.6f} "
      f"moy se={se.mean():.6f} sd vrai={sd_true:.6f} min={b1s.min():.4f} max={b1s.max():.4f}")
t = (b1s - beta1) / se
print("Couverture IC 95% :", np.mean(np.abs(t) < 1.9697))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(b1s, bins=60, density=True, color="C2", alpha=0.6, edgecolor="k", lw=0.3, label=r"$b_1^*$ simulés ($R=10\,000$)")
g = np.linspace(b1s.min(), b1s.max(), 300)
ax.plot(g, np.exp(-0.5 * ((g - 1) / sd_true) ** 2) / (sd_true * np.sqrt(2 * np.pi)), "C3", lw=2,
        label=r"Loi théorique $N(1,\ \sigma^2/\sum(x_i-\bar x)^2)$")
ax.axvline(b1s.mean(), color="k", ls="--", lw=1, label=f"Moyenne = {b1s.mean():.4f}")
ax.set_xlabel("$b_1^*$"); ax.set_ylabel("Densité"); ax.legend(fontsize=8)
ax.set_title("Distribution d'échantillonnage de $b_1$ (Monte Carlo)")
fig.tight_layout(); fig.savefig(out + "montecarlo.pdf"); plt.close(fig)

# biais MC avec erreurs non normales (khi-deux centré) : la moyenne reste 1
eps2 = (rng.chisquare(1, (R, N)) - 1) / np.sqrt(2)
B2 = (beta0 + beta1 * xm + eps2) @ Xm @ XtXi
print("MC erreurs khi2 centrées : moyenne b1 =", B2[:, 1].mean(), " ET =", B2[:, 1].std(ddof=1))

# ---------- 6. Générateur congruentiel linéaire ----------
def lcg(x0, a, lam, m, T):
    xs, us = [], []
    xt = x0
    for _ in range(T):
        xt = (a + lam * xt) % m
        xs.append(xt); us.append(xt / m)
    return xs, us
xs, us = lcg(7, 3, 5, 16, 20)
print("LCG jouet:", xs)
_, u = lcg(12345, 12345, 1103515245, 2 ** 31, 4000)
u = np.array(u)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4))
axs[0].hist(u, bins=20, density=True, color="C0", alpha=0.7, edgecolor="k", lw=0.3)
axs[0].set_title("4000 tirages $u_t$ d'un GCL"); axs[0].set_xlabel("$u$")
ex = -np.log(1 - u) / 2.0
axs[1].hist(ex, bins=40, density=True, color="C1", alpha=0.7, edgecolor="k", lw=0.3, label=r"$X=-\ln(1-U)/2$")
gg = np.linspace(0, ex.max(), 200); axs[1].plot(gg, 2 * np.exp(-2 * gg), "k", label=r"densité $2e^{-2x}$")
axs[1].set_title("Transformation inverse : loi exponentielle"); axs[1].legend(fontsize=8); axs[1].set_xlabel("$x$")
fig.tight_layout(); fig.savefig(out + "lcg.pdf"); plt.close(fig)
