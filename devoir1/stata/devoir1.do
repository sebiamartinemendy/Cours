*==============================================================================
* ECN-7025 Économétrie I - Devoir 1 (automne 2026)
* Questions 3 (Monte Carlo) et 4 (rendement de l'éducation)
*
* Placer ce fichier dans le même dossier que assign2School.raw et
* dictionary.raw, puis dans Stata :   do devoir1.do
*==============================================================================

version 14
clear all
set more off
capture log close
log using devoir1.log, replace text

*==============================================================================
* QUESTION 3 - Étude de Monte Carlo
*   y = 2 + 1*x1 + eps,  eps ~ N(0,1),  n = 250,  R = 1000,  seed 1234
*==============================================================================

set seed 1234

scalar beta0 = 2
scalar beta1 = 1
scalar sigma = 1
scalar gamma = 0.5
global R = 1000
global N = 250

set obs $N

* Régresseurs générés UNE SEULE FOIS : ils restent fixes d'une répétition
* à l'autre (propriétés conditionnelles à X, comme dans le cours)
generate double x1 = rnormal(0,1)
generate double x2 = gamma*x1 + rnormal(0,1)   // variable non pertinente (b)

correlate x1 x2

* Partie systématique du vrai modèle (x2 n'y figure pas : beta2 = 0)
generate double my = beta0 + beta1*x1
generate double y  = .

* Écarts-types théoriques sigma*sqrt([(X'X)^-1]_kk) pour les deux modèles
matrix accum XAXA = x1
matrix VA = sigma^2 * syminv(XAXA)
matrix accum XBXB = x1 x2
matrix VB = sigma^2 * syminv(XBXB)
display "E.-t. théorique b1 (modèle correct)      : " sqrt(VA[1,1])
display "E.-t. théorique b0 (modèle correct)      : " sqrt(VA[2,2])
display "E.-t. théorique b1 (modèle avec x2)      : " sqrt(VB[1,1])
display "E.-t. théorique b2 (modèle avec x2)      : " sqrt(VB[2,2])

tempname sim
postfile `sim' b0_a b1_a se0_a se1_a  b0_b b1_b b2_b se1_b se2_b  p2 rejet ///
    using q3_resultats, replace

quietly {
    forvalues r = 1/$R {
        * nouvel échantillon : nouvelles erreurs, mêmes x
        replace y = my + rnormal(0, sigma)

        * (a) modèle correct
        regress y x1
        scalar c_b0a  = _b[_cons]
        scalar c_b1a  = _b[x1]
        scalar c_se0a = _se[_cons]
        scalar c_se1a = _se[x1]

        * (b) modèle avec la variable non pertinente x2 (même y)
        regress y x1 x2
        scalar c_b0b  = _b[_cons]
        scalar c_b1b  = _b[x1]
        scalar c_b2b  = _b[x2]
        scalar c_se1b = _se[x1]
        scalar c_se2b = _se[x2]

        * (c) test de H0 : beta2 = 0 au seuil de 5 %
        test x2 = 0
        scalar c_p2 = r(p)

        post `sim' (c_b0a) (c_b1a) (c_se0a) (c_se1a) ///
                   (c_b0b) (c_b1b) (c_b2b) (c_se1b) (c_se2b) ///
                   (c_p2) (c_p2 < 0.05)
    }
}
postclose `sim'

use q3_resultats, clear

display _newline "===== Q3 (a) : modèle correct ====="
summarize b0_a b1_a se0_a se1_a
* Test de l'absence de biais : la moyenne des R estimations = vraie valeur ?
ttest b0_a == 2
ttest b1_a == 1

display _newline "===== Q3 (b) : avec la variable non pertinente x2 ====="
summarize b0_b b1_b b2_b se1_b se2_b
ttest b0_b == 2
ttest b1_b == 1
ttest b2_b == 0
* Perte d'efficacité : rapport des variances de b1
quietly summarize b1_a
scalar var_a = r(Var)
quietly summarize b1_b
scalar var_b = r(Var)
display "Var(b1 avec x2) / Var(b1 correct) = " var_b/var_a

display _newline "===== Q3 (c) : rejets de H0 : beta2 = 0 (alpha = 0.05) ====="
tabulate rejet
summarize rejet
* Intervalle de confiance de la proportion de rejets (doit contenir 0.05)
ci proportions rejet

* Graphiques
twoway (kdensity b1_a) (kdensity b1_b), xline(1)                        ///
    legend(order(1 "modèle correct" 2 "avec x2")) xtitle("b1")          ///
    title("Q3 : distribution de b1") name(q3_b1, replace)
graph export q3_b1.png, replace width(1600)
histogram p2, width(0.05) start(0) xline(0.05)                           ///
    title("Q3 (c) : p-values du test de beta2 = 0") name(q3_p, replace)
graph export q3_pvalues.png, replace width(1600)

*==============================================================================
* QUESTION 4 - Rendement de l'éducation (données de Card)
*==============================================================================

clear
infile using dictionary.raw, using(assign2School.raw) clear
describe, short
summarize lwage educ exper south

* ---------- (a) Régression de base ----------
regress lwage educ exper
display "Rendement exact de l'éducation   : " 100*(exp(_b[educ])-1)  " %"
display "Rendement exact de l'expérience  : " 100*(exp(_b[exper])-1) " %"

* ---------- (b) Moyenne des résidus ----------
predict double uhat, residuals
summarize uhat

* ---------- (c) Test H0 : beta_educ = 0.07 (test t « à la main ») ----------
quietly regress lwage educ exper
scalar t_c = (_b[educ] - 0.07)/_se[educ]
scalar p_c = 2*ttail(e(df_r), abs(t_c))
scalar crit = invttail(e(df_r), 0.025)
display "t = " t_c "   p-value = " p_c "   valeur critique 5 % = " crit
* Vérifications équivalentes
test educ = 0.07
lincom educ - 0.07

* ---------- (d) Reparamétrisation ----------
* lwage = b0 + (theta + 0.07)*educ + b2*exper  =>  lwage - 0.07*educ = b0 + theta*educ + ...
generate double lwage_r = lwage - 0.07*educ
regress lwage_r educ exper
* -> la ligne educ donne directement t et la p-value du test de (c)

* ---------- (e) Différence constante Sud / ailleurs ----------
regress lwage educ exper south
display "Écart salarial exact Sud : " 100*(exp(_b[south])-1) " %"
test south
regress lwage educ exper south, vce(robust)

* ---------- (f) Test de Chow : marchés identiques ? ----------
regress lwage c.educ##i.south c.exper##i.south
scalar ssr_nc = e(rss)
scalar df_nc  = e(df_r)
test 1.south 1.south#c.educ 1.south#c.exper
* Test des seules pentes
test 1.south#c.educ 1.south#c.exper

* Chow « à la main » : SCE contrainte (modèle (a)) vs SCE non contrainte
quietly regress lwage educ exper
scalar ssr_c = e(rss)
scalar F_chow = ((ssr_c - ssr_nc)/3) / (ssr_nc/df_nc)
display "SCE_C = " ssr_c "  SCE_NC = " ssr_nc "  F = " F_chow ///
        "  p = " Ftail(3, df_nc, F_chow) "  crit 5 % = " invFtail(3, df_nc, 0.05)

* Vérification : SCE_NC = somme des SCE des régressions séparées
quietly regress lwage educ exper if south == 1
scalar ssr_s = e(rss)
regress lwage educ exper if south == 1
quietly regress lwage educ exper if south == 0
scalar ssr_n = e(rss)
regress lwage educ exper if south == 0
display "SCE Sud + SCE ailleurs = " ssr_s + ssr_n

* Version robuste à l'hétéroscédasticité
regress lwage c.educ##i.south c.exper##i.south, vce(robust)
test 1.south 1.south#c.educ 1.south#c.exper

* Écart Sud - ailleurs selon le profil (educ = 12 et 16, exper = 8)
quietly regress lwage c.educ##i.south c.exper##i.south
lincom 1.south + 12*1.south#c.educ + 8*1.south#c.exper
lincom 1.south + 16*1.south#c.educ + 8*1.south#c.exper

log close
