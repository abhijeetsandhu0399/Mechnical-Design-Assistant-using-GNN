"""Create human-readable metrics, examples, and learning/parity plots."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanics import NAMES,INPUTS,UNITS

def make_report(out):
    out=Path(out); m=json.loads((out/'metrics.json').read_text())
    d=np.load(out/'test_predictions.npz'); y,p=d['truth'],d['prediction']
    h=np.genfromtxt(out/'history.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.semilogy(h['epoch']+1,h['loss'],label='Training'); ax.semilogy(h['epoch']+1,h['val_loss'],label='Validation')
    ax.set(xlabel='Epoch',ylabel='MSE of standardized targets',title='Mechanical Design Assistant: learning curves'); ax.legend(); fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,i in zip(axes.flat,[7,8,10,11,12,17]):
        ax.scatter(y[:,i],p[:,i],s=3,alpha=.3)
        lo,hi=y[:,i].min(),y[:,i].max(); ax.plot([lo,hi],[lo,hi],color='black',lw=1)
        ax.set(title=NAMES[i],xlabel=f'Analytical ({UNITS[i]})',ylabel=f'GNN prediction ({UNITS[i]})')
    fig.suptitle('Held-out test set: raw GNN outputs'); fig.tight_layout(); fig.savefig(out/'test_parity.png',dpi=160); plt.close(fig)
    lines=['# Measured results','',f"{m['config']['samples']:,}-case experiment; actual split counts: {m['counts']}.",
           f"Best epoch: {m['best_epoch']} of {m['epochs_run']}; {m['parameters']:,} trainable parameters.",
           '', 'R2 below is the macro average over nonconstant outputs; constant-output R2 is undefined.',
           '', '| Model / partition | Mean R2 | Standardized RMSE |','|---|---:|---:|']
    for k in ['validation','test','mean_baseline','linear_baseline','out_of_distribution']:
        lines.append(f"| {k} | {m[k]['mean_r2']:.6f} | {m[k]['standardized_rmse']:.6f} |")
    lines+=['','## Test errors in physical units','','| Output | Unit | MAE | RMSE | R2 |','|---|---|---:|---:|---:|']
    for row in m['test']['per_output']:
        r2=f"{row['r2']:.6f}" if row['r2'] is not None else 'undefined (constant)'
        lines.append(f"| {row['output']} | {row['unit']} | {row['mae']:.6f} | {row['rmse']:.6f} | {r2} |")
    lines+=['','## Equilibrium residuals of raw GNN predictions','','| Body | Force RMS (N) | Moment RMS (N m) | Max abs force (N) | Max abs moment (N m) |','|---|---:|---:|---:|---:|']
    for k,v in m['test']['equilibrium'].items():
        lines.append(f"| {k} | {v['force_rms_N']:.6f} | {v['moment_rms_Nm']:.6f} | {v['force_max_abs_N']:.6f} | {v['moment_max_abs_Nm']:.6f} |")
    lines+=['','Analytical labels satisfy all three body balances to floating-point precision. GNN outputs are not projected onto equilibrium.',
            'Errors measure agreement with synthetic rigid-body equations, not experimental validation or strength/fatigue safety.',
            'The out-of-distribution set doubles applied loads and scales lengths by 1.4; it is a stress test, not a promised operating range.',
            '', '## Example held-out cases']
    for j in range(3):
        lines+=['',f"### Test example {j+1} (dataset ID {d['ids'][j]})",'',str(dict(zip(INPUTS,np.round(d['x'][j],6).tolist()))),'',
                '| Quantity | Analytical | Predicted |','|---|---:|---:|']
        for i in [6,7,8,10,11,12,17]: lines.append(f'| {NAMES[i]} | {y[j,i]:.5f} | {p[j,i]:.5f} |')
    lines+=['','![Learning curves](learning_curves.png)','','![Test parity](test_parity.png)']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    import sys
    make_report(sys.argv[1] if len(sys.argv)>1 else 'artifacts')
