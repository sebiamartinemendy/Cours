"""Question 3 : étude de Monte Carlo (réplique Python du do-file Stata)."""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Serif"})
rng = np.random.default_rng(1234)
b0, b1, sigma, R, n, gamma = 2.0, 1.0, 1.0, 1000, 250, 0.5
x1 = rng.normal(0, 1, n)                     # fixe d'une répétition à l'autre
x2 = gamma * x1 + rng.normal(0, 1, n)        # variable non pertinente, fixe
XA = np.column_stack([np.ones(n), x1]); XB = np.column_stack([np.ones(n), x1, x2])
def ols(X, y):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1]); return b, np.sqrt(s2 * np.diag(XtXi))
resA, resB = [], []
for r in range(R):
    y = b0 + b1 * x1 + rng.normal(0, sigma, n)
    bA, seA = ols(XA, y); bB, seB = ols(XB, y)
    resA.append(np.r_[bA, seA]); resB.append(np.r_[bB, seB])
A = np.array(resA); B = np.array(resB)
tcrit = stats.t.ppf(0.975, n - 3)
t2 = B[:, 2] / B[:, 5]; rej = np.abs(t2) > tcrit
p2 = 2 * stats.t.sf(np.abs(t2), n - 3)
sd_true_A = sigma * np.sqrt(np.diag(np.linalg.inv(XA.T @ XA)))
sd_true_B = sigma * np.sqrt(np.diag(np.linalg.inv(XB.T @ XB)))
print("corr(x1,x2) =", np.corrcoef(x1, x2)[0, 1])
def row(name, v): print(f"{name:14s} moy={v.mean():.4f}  et={v.std(ddof=1):.4f}  min={v.min():.4f}  max={v.max():.4f}")
print("--- (a) modèle correct"); row("b0", A[:, 0]); row("b1", A[:, 1]); row("se(b0)", A[:, 2]); row("se(b1)", A[:, 3])
print("   e.t. théoriques", sd_true_A, " erreur MC b1:", A[:, 1].std(ddof=1) / np.sqrt(R))
print("   t-test biais b0:", stats.ttest_1samp(A[:, 0], 2), " b1:", stats.ttest_1samp(A[:, 1], 1))
print("--- (b) avec x2 non pertinente"); row("b0", B[:, 0]); row("b1", B[:, 1]); row("b2", B[:, 2]); row("se(b1)", B[:, 4]); row("se(b2)", B[:, 5])
print("   e.t. théoriques", sd_true_B)
print("   ratio var b1 B/A empirique", B[:, 1].var(ddof=1) / A[:, 1].var(ddof=1), " théorique", (sd_true_B[1] / sd_true_A[1]) ** 2)
print("   t-test biais b0:", stats.ttest_1samp(B[:, 0], 2), " b1:", stats.ttest_1samp(B[:, 1], 1))
print("--- (c) taux de rejet H0: beta2=0 :", rej.mean(), " nb rejets", rej.sum(), " IC95 MC:", rej.mean() - 1.96 * np.sqrt(.05 * .95 / R), rej.mean() + 1.96 * np.sqrt(.05 * .95 / R))
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
g = np.linspace(0.8, 1.2, 200)
ax[0].hist(A[:, 1], bins=35, density=True, alpha=.55, color="C0", label="modèle correct")
ax[0].hist(B[:, 1], bins=35, density=True, alpha=.45, color="C1", label="avec $x_2$")
ax[0].axvline(1, color="k", ls="--"); ax[0].set_title("$b_1$ : centré sur 1 dans les 2 cas"); ax[0].legend(fontsize=7)
ax[1].hist(t2, bins=35, density=True, alpha=.6, color="C2"); gg = np.linspace(-4, 4, 200)
ax[1].plot(gg, stats.t.pdf(gg, n - 3), "k"); [ax[1].axvline(s * tcrit, color="r", ls="--") for s in (-1, 1)]
ax[1].set_title(f"$t$ de $H_0:\\beta_2=0$ (rejet : {rej.mean():.1%})")
ax[2].hist(p2, bins=20, range=(0, 1), density=True, alpha=.6, color="C4"); ax[2].axvline(.05, color="r", ls="--")
ax[2].set_title("$p$-values : loi uniforme sous $H_0$")
fig.tight_layout(); fig.savefig("figures/q3_montecarlo.pdf")
