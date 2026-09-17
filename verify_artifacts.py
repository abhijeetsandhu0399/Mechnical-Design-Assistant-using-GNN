"""Audit saved splits, normalization, metrics, and checkpoint inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from predict import predict
from mechanics import solve,residuals
from train import metrics

def verify(folder):
    folder=Path(folder)
    data=np.load(folder/'dataset.npz')
    norm=np.load(folder/'normalization.npz')
    result=np.load(folder/'test_predictions.npz')
    report=json.loads((folder/'metrics.json').read_text())
    ids=np.concatenate([data[name] for name in ['train','validation','test']])
    np.testing.assert_array_equal(np.sort(ids),np.arange(len(data['x'])))
    np.testing.assert_allclose(norm['mean'],data['y'][data['train']].mean(0))
    expected_std=data['y'][data['train']].std(0)
    np.testing.assert_allclose(norm['std'],np.where(expected_std<1e-8,1.,expected_std))
    np.testing.assert_allclose(solve(data['x']),data['y'],atol=1e-12)
    np.testing.assert_allclose(residuals(data['x'],data['y']),0,atol=1e-10)
    np.testing.assert_array_equal(result['ids'],data['test'])
    np.testing.assert_array_equal(result['x'],data['x'][data['test']])
    np.testing.assert_array_equal(result['truth'],data['y'][data['test']])
    recomputed=metrics(result['truth'],result['prediction'],norm['std'])
    np.testing.assert_allclose(recomputed['mean_r2'],report['test']['mean_r2'],atol=1e-12)
    reloaded=predict(result['x'][:5],folder)
    difference=float(np.max(np.abs(reloaded-result['prediction'][:5])))
    np.testing.assert_allclose(reloaded,result['prediction'][:5],atol=2e-4,rtol=1e-5)
    info={'status':'passed','samples':len(data['x']),'test_samples':len(result['x']),
          'max_checkpoint_reload_difference':difference,
          'checks':['split disjointness and coverage','train-only normalization','all analytical labels',
                    'all-body equilibrium','test IDs and inputs','recomputed mean R2','checkpoint inference reload']}
    (folder/'verification.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',nargs='?',default='artifacts')
    verify(parser.parse_args().folder)
