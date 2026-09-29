*==============================================================================
* Chapitre 2 - Les moindres carrés ordinaires
* Étude de Monte Carlo : propriétés de l'estimateur MCO
*
* Modèle simulé :  y_i = 1 + x_i + eps_i ,   eps_i ~ NID(0, 1)
*
* Partie 1 : réplication EXACTE du code du cours (même graine, même ordre de
*            tirages, mêmes types de variables) -> doit redonner les résultats
*            des notes : moyenne de b = .999501, e.t. = .0343199,
*            moyenne des se = .0344057, sdbeta1 = .03444001
* Partie 2 : analyse complémentaire (biais, erreur de simulation, test,
*            couverture des intervalles de confiance, graphique)
* Partie 3 : erreurs NON normales (khi-deux centré) : toujours sans biais
* Partie 4 : violation de E[eps|x] = 0 : l'estimateur devient biaisé
*
* Utilisation : ouvrir dans Stata (version 14 ou plus récente), puis
*               do montecarlo_mco.do
* Durée : environ 1 à 3 minutes (3 x 10 000 régressions).
*==============================================================================

version 14
clear all
set more off
capture log close
log using montecarlo_mco.log, replace text

*------------------------------------------------------------------------------
* PARTIE 1 - Réplication du code du cours
*------------------------------------------------------------------------------

* Graine : rend les tirages (donc les résultats) reproductibles
set seed 12345678

* Vraies valeurs des paramètres (le « monde » que l'on fabrique)
scalar beta0 = 1
scalar beta1 = 1
scalar sigma = 1

* R : nombre de réplications ; n : taille de chaque échantillon
global Nreps = 10000
global Nobs  = 250
set obs $Nobs

* x log-normal, tiré UNE SEULE FOIS : il reste fixe d'une réplication à l'autre
* (on étudie les propriétés conditionnelles à X, comme dans la théorie)
generate double x = exp(rnormal(0,1))

* Vraie variance de la pente : Var(b1|X) = sigma^2 / somme (x_i - xbar)^2
* r(Var) divise par n-1, donc r(Var)*(n-1) = somme des carrés des écarts.
* (Le cours écrit sigma au lieu de sigma^2 : identique ici car sigma = 1.)
summarize x
scalar varbeta1 = sigma^2/( r(Var)*($Nobs-1) )
scalar sdbeta1  = sqrt(varbeta1)

* Partie systématique E[y|x] (ne change pas d'une réplication à l'autre)
* NB : types « float » comme dans le cours, pour reproduire ses chiffres exacts
generate my      = beta0 + beta1*x
generate epsilon = .
generate sy      = .

* Fichier qui recevra, pour chaque réplication, la pente et son écart-type
tempname simulate
postfile `simulate' b se using simresults, replace

quietly {
    forvalues i = 1/$Nreps {
        replace epsilon = rnormal(0, sigma)   // nouvelles erreurs (2e arg. = écart-type)
        replace sy = my + epsilon             // nouvel échantillon y*
        regress sy x                          // estimation MCO
        post `simulate' (_b[x]) (_se[x])      // sauvegarde de b1* et se(b1*)
    }
}
postclose `simulate'

*------------------------------------------------------------------------------
* PARTIE 3 - Erreurs non normales : eps = (khi2(1) - 1)/sqrt(2)
*            (moyenne 0, variance 1, mais très asymétriques)
*            Faite AVANT d'ouvrir les résultats car on a encore besoin de x.
*------------------------------------------------------------------------------
tempname simchi2
postfile `simchi2' b se using simresults_chi2, replace
quietly {
    forvalues i = 1/$Nreps {
        replace epsilon = sigma*(rchi2(1) - 1)/sqrt(2)
        replace sy = my + epsilon
        regress sy x
        post `simchi2' (_b[x]) (_se[x])
    }
}
postclose `simchi2'

*------------------------------------------------------------------------------
* PARTIE 4 - Violation de l'exogénéité : E[eps|x] = 0.2*x
*            eps = 0.2*x + u, u ~ N(0,1)  =>  y = 1 + 1.2*x + u
*------------------------------------------------------------------------------
tempname simendo
postfile `simendo' b se using simresults_endo, replace
quietly {
    forvalues i = 1/$Nreps {
        replace epsilon = 0.2*x + rnormal(0, sigma)
        replace sy = my + epsilon
        regress sy x
        post `simendo' (_b[x]) (_se[x])
    }
}
postclose `simendo'

*------------------------------------------------------------------------------
* PARTIE 1 (suite) - Résultats de la réplication du cours
*------------------------------------------------------------------------------
use simresults, clear

display _newline "=== Partie 1 : résultats (à comparer aux notes, p. 27) ==="
summarize
scalar list sdbeta1

*------------------------------------------------------------------------------
* PARTIE 2 - Analyse complémentaire
*------------------------------------------------------------------------------
display _newline "=== Partie 2 : analyse complémentaire ==="

* (a) Biais estimé et erreur de simulation (e.t. de la moyenne = sd/sqrt(R))
quietly summarize b
display "Moyenne des b1*             : " %9.6f r(mean)
display "Biais estimé  (moy. - 1)    : " %9.6f r(mean) - beta1
display "Erreur de simulation        : " %9.6f r(sd)/sqrt(r(N))
display "E.t. empirique des b1*      : " %9.6f r(sd)
display "E.t. théorique (sdbeta1)    : " %9.6f sdbeta1

quietly summarize se
display "Moyenne des e.t. estimés    : " %9.6f r(mean)

* (b) Test formel H0 : E[b1] = 1 (absence de biais)
ttest b == 1

* (c) Couverture de l'intervalle de confiance à 95 % : b1 +/- t(0.975; n-2)*se
*     Doit contenir la vraie valeur dans environ 95 % des échantillons
scalar tcrit = invttail($Nobs - 2, 0.025)
generate byte couvre = abs((b - beta1)/se) < tcrit
display _newline "Taux de couverture de l'IC à 95 % :"
summarize couvre

* (d) Histogramme + densité normale théorique N(1, sdbeta1^2)
local sd = sdbeta1
histogram b, density normal                                                ///
    title("Distribution d'échantillonnage de b1 (R = 10 000, n = 250)")     ///
    xtitle("b1*") name(mc_normal, replace)
graph export mc_hist_normal.png, replace width(1600)

twoway (histogram b, density color(eltgreen))                              ///
       (function y = normalden(x, 1, `sd'), range(0.85 1.15) lcolor(red)), ///
       legend(order(1 "b1* simulés" 2 "Loi théorique N(1, sdbeta1²)"))     ///
       title("Histogramme simulé vs loi théorique") xtitle("b1*")          ///
       name(mc_theorie, replace)
graph export mc_hist_theorie.png, replace width(1600)

*------------------------------------------------------------------------------
* PARTIE 3 (suite) - Erreurs non normales
*------------------------------------------------------------------------------
use simresults_chi2, clear
display _newline "=== Partie 3 : erreurs khi-deux centrées (non normales) ==="
summarize b se
ttest b == 1
histogram b, density normal                                                ///
    title("Erreurs non normales : b1* reste centré sur 1, quasi normal")   ///
    xtitle("b1*") name(mc_chi2, replace)
graph export mc_hist_chi2.png, replace width(1600)

*------------------------------------------------------------------------------
* PARTIE 4 (suite) - Violation de l'exogénéité
*------------------------------------------------------------------------------
use simresults_endo, clear
display _newline "=== Partie 4 : E[eps|x] = 0.2x  (biais attendu : +0.2) ==="
summarize b se
ttest b == 1
histogram b, density                                                       ///
    title("Exogénéité violée : b1* centré sur 1.2, pas sur 1")             ///
    xtitle("b1*") xline(1, lcolor(red)) name(mc_endo, replace)
graph export mc_hist_endo.png, replace width(1600)

log close
