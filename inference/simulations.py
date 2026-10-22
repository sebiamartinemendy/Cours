"""Simulations de Monte Carlo pour les exercices d'inférence (ex. 4, 6, 7) et calculs numériques."""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Serif"})

def ols(X, y):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    n, k = X.shape; s2 = e @ e / (n - k)
    V = s2 * XtXi
    meat = (X * e[:, None] ** 2).T @ X
    Vw = XtXi @ meat @ XtXi
    return b, np.sqrt(np.diag(V)), np.sqrt(np.diag(Vw))

print("=== Ex1 inférence")
se = np.sqrt(1**2 * 0.04); print("se", se, "IC", 5 - 1.96 * se, 5 + 1.96 * se, "t", 5 / se, "p", 2 * stats.norm.sf(25))
print("=== Ex2", np.sqrt(0.04 + 0.03 + 2 * 0.01))
print("=== Ex9")
for df in [1196]:
    t = stats.t.isf(0.053 / 2, df); print("t exper", t, "se", 0.0298776 / t)
    c = stats.t.ppf(.975, df); print("crit", c, "Fcrit", stats.f.ppf(.95, 1, df))
    se3 = (0.4018721 - 0.3038663) / (2 * c); print("se3", se3, "t3", -0.3528692 / se3, "mid", (-0.4018721 - 0.3038663) / 2)
    print("hours t", .0756342 / .0067843, "IC", .0756342 - c * .0067843, .0756342 + c * .0067843)
print(np.exp(.0756342) - 1, np.exp(-.3528692) - 1)

print("=== Ex4 Monte Carlo (R=1000)")
rng = np.random.default_rng(2024); R = 1000
rows = []
for n in [25, 50, 100, 200]:
    x = rng.normal(0, 1, n)          # x fixe pour les R répétitions
    X = np.column_stack([np.ones(n), x])
    B = np.array([np.linalg.lstsq(X, 1 + 0.8 * x + rng.normal(0, 1, n), rcond=None)[0] for _ in range(R)])
    m = B.mean(0); sd = B.std(0, ddof=1)
    rows.append((n, m[0], m[1], m[0] - 1, m[1] - 0.8, sd[0], sd[1], sd[1] / np.sqrt(R)))
    print(n, "moy", m, "biais", m - [1, .8], "et", sd, "err MC b2", sd[1] / np.sqrt(R))
np.save("ex4_rows.npy", np.array(rows))
ns = [r[0] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].axhline(0, color="k", lw=.8)
ax[0].errorbar(ns, [r[3] for r in rows], yerr=[1.96 * r[5] / np.sqrt(R) for r in rows], fmt="o-", label=r"biais de $\hat\beta_1$", capsize=3)
ax[0].errorbar(ns, [r[4] for r in rows], yerr=[1.96 * r[7] for r in rows], fmt="s-", label=r"biais de $\hat\beta_2$", capsize=3)
ax[0].set_xlabel("n"); ax[0].set_title("Biais estimé (± 1,96 erreur MC)"); ax[0].legend(fontsize=8)
ax[1].plot(ns, [r[5] for r in rows], "o-", label=r"é.-t. de $\hat\beta_1$"); ax[1].plot(ns, [r[6] for r in rows], "s-", label=r"é.-t. de $\hat\beta_2$")
ax[1].set_xlabel("n"); ax[1].set_title("Écart-type : diminue comme $1/\\sqrt{n}$"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/ex4_biais.pdf")

print("=== Ex6 : variable omise + hétéroscédasticité")
b0, b1, b2, eta = 1, 1, 0.5, 0.5
for n in [100, 1000]:
    x1 = rng.normal(0, 1, n); u = rng.normal(0, 1, n); x2 = eta * x1 + u
    sig = np.sqrt(0.5 + x1 ** 2)   # hétéroscédasticité : Var(eps|x)=0.5+x1^2
    Xs = np.column_stack([np.ones(n), x1]); Xf = np.column_stack([np.ones(n), x1, x2])
    bs, ses, sws, rej, rejc = [], [], [], [], []
    for _ in range(2000):
        y = b0 + b1 * x1 + b2 * x2 + sig * rng.normal(0, 1, n)
        b, s, w = ols(Xs, y); bs.append(b[1]); ses.append(s[1]); sws.append(w[1])
    bs = np.array(bs)
    print(n, "moy b1", bs.mean(), "attendu", b1 + b2 * np.polyfit(x1, x2, 1)[0], "et emp", bs.std(ddof=1), "moy se classique", np.mean(ses), "moy se White", np.mean(sws))
    # test de beta2 = 0 dans le modèle complet : sous H0 (b2=0) et sous H1 (b2=0.5)
    for bb2 in [0.0, 0.5]:
        rr, rc = [], []
        for _ in range(2000):
            y = b0 + b1 * x1 + bb2 * x2 + sig * rng.normal(0, 1, n)
            b, s, w = ols(Xf, y); c = stats.t.ppf(.975, n - 3)
            rr.append(abs(b[2] / w[2]) > c); rc.append(abs(b[2] / s[2]) > c)
        print("   vrai beta2 =", bb2, "rejet robuste", np.mean(rr), "rejet classique", np.mean(rc))

print("=== Ex7 : x corrélé avec eps")
alpha, beta = 0.0, 1.0
for mux in [0.0, 2.0]:
    for n in [100, 1000]:
        bs, rejs = [], []
        for _ in range(2000):
            z = rng.normal(0, 1, n); e = rng.normal(0, 1, n)
            x = mux + z + 0.5 * e          # Cov(x,eps)=0.5, Var(x)=1.25
            y = alpha + beta * x + e
            b, s, w = ols(np.column_stack([np.ones(n), x]), y)
            bs.append(b[1]); rejs.append(abs(b[0] / s[0]) > stats.t.ppf(.975, n - 2))
        print("mux", mux, "n", n, "moy b", np.mean(bs), "plim theo", beta + 0.5 / 1.25, "rejet alpha=0", np.mean(rejs))
