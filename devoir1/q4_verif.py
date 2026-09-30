import wooldridge as w, numpy as np, statsmodels.formula.api as smf
from scipy import stats
d=w.data('card')
m=smf.ols('lwage~educ+exper',d).fit(); print(m.summary().tables[1]); print('n',m.nobs,'R2',m.rsquared,'s',np.sqrt(m.scale),'df',m.df_resid)
print('mean resid',m.resid.mean(), 'sum',m.resid.sum())
b=m.params['educ']; se=m.bse['educ']; t=(b-0.07)/se; p=2*stats.t.sf(abs(t),m.df_resid)
print('t',t,'p',p, 'crit',stats.t.ppf(.975,m.df_resid))
print(m.t_test('educ=0.07'))
d['y2']=d.lwage-0.07*d.educ
m2=smf.ols('y2~educ+exper',d).fit(); print(m2.summary().tables[1])
me=smf.ols('lwage~educ+exper+south',d).fit(); print(me.summary().tables[1]); print('R2',me.rsquared,'SSR',me.ssr)
print('exp(b)-1',np.exp(me.params.south)-1)
# f: Chow
mf=smf.ols('lwage~(educ+exper)*south',d).fit(); print(mf.summary().tables[1]); print('SSR',mf.ssr,'df',mf.df_resid)
print(mf.f_test('south=0, educ:south=0, exper:south=0'))
ssr_r=m.ssr; ssr_u=mf.ssr; F=((ssr_r-ssr_u)/3)/(ssr_u/mf.df_resid); print('SSR_R',ssr_r,'SSR_U',ssr_u,'F',F,'p',stats.f.sf(F,3,mf.df_resid),'crit',stats.f.ppf(.95,3,mf.df_resid))
s=smf.ols('lwage~educ+exper',d[d.south==1]).fit(); n=smf.ols('lwage~educ+exper',d[d.south==0]).fit()
print('south',s.params.values,s.ssr,s.nobs,' nonsouth',n.params.values,n.ssr,n.nobs, 'sum',s.ssr+n.ssr)
print(mf.f_test('educ:south=0, exper:south=0'))
# robust
print(mf.get_robustcov_results('HC1').f_test('south=0, educ:south=0, exper:south=0'))
print(me.get_robustcov_results('HC1').summary().tables[1])
print('n south', d.south.sum())
