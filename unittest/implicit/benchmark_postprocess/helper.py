import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def translate_label(label):
    if label == 'K':
        return 'K'
    elif label == 'nx':
        return r'$n_x$'
    elif label == 'nu':
        return r'$n_u$'
    elif label == 'ng':
        return r'$n_g$'
    elif label == 'r':
        return r'$r$'
    elif label == 'ng_ineq':
        return r'$n_{g,ineq}$'
    else:
        raise ValueError(f'Unknown label: {label}')
    

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 14,
              'axes.titlesize': 15,
              'legend.fontsize': 10,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [4, 3],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)
    
def get_data():
    file = '../../../build_docker/random_benchmark_results_extended_20000.csv'
    df = pd.read_csv(file)

    data = {
        'K': np.array(df['K'].values),
        'nx': np.array(df['nx'].values),
        'r': np.array(df['r'].values),
        'nu': np.array(df['nu'].values),
        'ng': np.array(df['ng'].values),
        'ng_ineq': np.array(df['ng_ineq'].values),
        't_expl': np.array(df['t_expl'].values),
        't_impl': np.array(df['t_impl'].values),
        't_impl_pre': np.array(df['t_impl_pre'].values),
        't_impl_solve': np.array(df['t_impl_solve'].values),
        't_impl_post': np.array(df['t_impl_post'].values),
        't_reform': np.array(df['t_reform'].values),
        't_accel': np.array(df['t_accel'].values),
        'lu_expl': np.array(df['lu_expl'].values),
        'lu_impl': np.array(df['lu_impl'].values),
        'lu_reform': np.array(df['lu_reform'].values),
        'lu_accel': np.array(df['lu_accel'].values),
        'impl_decomp': np.array(df['impl_decomp'].values),
    }
    
    df['nz'] = df['nx']
    df['nf'] = df['nx']
    
    # number of rows: ng + nf
    df['m'] = df['ng'] + df['nf']
    df['m_rel'] = df['ng'] / df['m']
    
    # number of cols: nu + nz
    df['n'] = df['nx'] + df['nu']
    df['n_rel'] = df['nu'] / df['n']

    return df