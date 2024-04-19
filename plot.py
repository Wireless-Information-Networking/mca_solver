import numpy as np
import pandas as pd
import seaborn as sns

pd.options.display.max_rows = None
pd.options.display.max_columns = 120
# %config InlineBackend.figure_dpi = [600]
# %config InlineBackend.figure_formats = ['png']

st = pd.read_csv('results/res_1689862982792443575.csv', delimiter=',\t', engine='python')


# ---
sns.set(font_scale=1., rc={'text.usetex' : True, 'lines.markersize': 6., 'legend.markerscale': 1.0})  # , 'figure.figsize': (5, 2)})
filt = (st.time < 10e9) & (st.scale != -5)

st2 = st.drop(columns=['weights', 'input', 'error'])[filt].copy()
st2.type = st2.type.apply(lambda x: ' (schur)' if x == 'shur' else '')
st2.method = st2.method.apply(lambda x: 'proposed approach' if x == 'ours' else x) + st2.type

st2_avg = st2.groupby(['method', 'type', 'scale', 'size', 'pcond', 'iter']).mean()  # , 'time', 'norm'


# ---
time_select = np.linspace(0, 10e9, 10)


def time_agg(xx):
    x = xx.reset_index()
    a = x['time'] < -1
    for t in time_select:
        a |= (x.index == (x['time'] - t).abs().idxmin())
    a = x.loc[a]
    return [a.time.to_list(), a.norm.to_list()]


st3 = st2_avg.droplevel(-1).groupby(level=[0, 1, 2, 3, 4]).apply(time_agg, include_groups=False)

st3 = pd.DataFrame({'time': st3.apply(lambda x: x[0]), 'norm': st3.apply(lambda x: x[1])}, index=st3.index)
st3 = st3.reset_index(level=['method', 'type', 'scale', 'size', 'pcond'])
st3 = st3.explode(['time', 'norm'])

# ---
gg = sns.relplot(data=st2_avg, x='time', y='norm', hue='method', col='scale', row='size', style='pcond', kind='line',  # units='weights',
                 facet_kws={'sharey': 'row'}, height=2, aspect=2, dashes=True, markers=True, markevery=15000)  # , kind='scatter', s=20
gg.set_axis_labels('Time (ns)', '')  # '$||\\mathbf{i}^{ext} - \\mathbf{G}_{ABCD}\\mathbf{v}^{ext}||_F$'
gg.set_titles(col_template='$\\alpha = 2^{{ {col_name} }}$', row_template='{row_name} array')
gg.set(yscale='log', xlim=(0, 10**10))  # log_10

for txt, ch in zip(gg.legend.get_texts(), ('\\textbf{Method:}', 'GMRES', 'GMRES (Schur)', 'LGMRES', 'LGMRES (Schur)', 'Ours', '\\textbf{Preconditioner:}', 'None or N/A', 'Jacobi', '[9]'), ):
    txt.set_text(ch)

for en, axs in enumerate(gg.axes):
    for em, ax in enumerate(axs):
        st4 = st3[(st3.scale == -11 + 2*em) & (st3['size'] == ['128x128', '256x256', '512x512'][en])]
        sns.scatterplot(data=st4, x='time', y='norm', hue='method', style='pcond', markers=True, legend=False, ax=ax)
        if em == 0:
            if en == 1:
                ax.set_ylabel('$\\log_{10}~Avg.~||\\mathbf{i}^{ext} - \\mathbf{G}_{ABCD}\\mathbf{v}^{ext}||_F$')
            else:
                ax.set_ylabel('')
        ax.axhline(1e-5, lw=.5, ls='-.', c='#888888')

# ---
gg.savefig('norm_time.png', dpi=600)
print('fin')
