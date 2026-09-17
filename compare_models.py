"""Matched-information MLP comparison and resident CPU inference benchmark.

Run from the project folder: python compare_models.py --out comparison
Uses shared split indices and a fixed GNN checkpoint for evaluation.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from graph_data import tf, tfgnn, SOURCE, TARGET, features
from mechanics import solve, validate
from model import build_model
from train import metrics, residual_metrics

SCALE = np.array([100, 100, 100, .1, .1, .1, .1], np.float64)

def mlp_model(seed):
    layers = [tf.keras.layers.Input(shape=(7,))]
    for i, width in enumerate([208, 208, 180, 19]):
        layers.append(tf.keras.layers.Dense(
            width, activation='swish' if i < 3 else None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+i)))
    return tf.keras.Sequential(layers)

def fast_graph(x):
    """Vectorized, inference-only version of graph_data.features; same features."""
    x = validate(x)
    b = len(x)
    nodes = np.zeros((b, 4, 7), np.float32)
    nodes[:, :, :4] = np.eye(4)
    nodes[:, 0, 4] = x[:, 3] / .1
    nodes[:, 1, 4] = x[:, 4] / .2
    nodes[:, 2, 6] = x[:, 6] / .1
    points = np.zeros((b, 6, 3))
    points[:, 0, 0] = points[:, 1, 0] = x[:, 3]
    points[:, 0, 2] = x[:, 5]
    points[:, 3, 0] = points[:, 4, 0] = x[:, 4]
    points[:, 5, 2] = x[:, 6]
    edges = np.zeros((b, 12, 14), np.float32)
    edges[:, :, :3] = np.repeat(points / .1, 2, axis=1)
    edges[:, :, 3:9] = np.repeat(np.eye(6), 2, axis=0)
    edges[:, 0, 9:12] = x[:, :3] / 100
    edges[:, 1, 9:12] = -x[:, :3] / 100
    edges[:, :2, 12] = 1
    edges[:, :, 13] = np.tile([1., -1.], 6)
    return tfgnn.GraphTensor.from_pieces(
        node_sets={'bodies': tfgnn.NodeSet.from_fields(
            sizes=np.full((b, 1), 4, np.int32), features={'features':nodes})},
        edge_sets={'interactions':tfgnn.EdgeSet.from_fields(
            sizes=np.full((b, 1), 12, np.int32), features={'features':edges},
            adjacency=tfgnn.Adjacency.from_indices(
                source=('bodies', np.broadcast_to(SOURCE,(b,12)).copy()),
                target=('bodies', np.broadcast_to(TARGET,(b,12)).copy())))})

def predict_mlp(model, x, mean, std):
    return model(np.asarray(x / SCALE,np.float32),training=False).numpy()*std+mean

def score(x, y, pred, std):
    out=metrics(y,pred,std)
    out['equilibrium']=residual_metrics(x,pred)
    out['bearing_mae_N']=float(np.abs(pred[:,[6,7,8,10,11]]-y[:,[6,7,8,10,11]]).mean())
    return out

def main(args):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    art=Path(args.artifacts)
    data=np.load(art/'dataset.npz'); x,y=data['x'],data['y']
    norm=np.load(art/'normalization.npz'); mean,std=norm['mean'],norm['std']
    tr,va,te=data['train'],data['validation'],data['test']
    assert len(set(tr)&set(te))==0 and len(set(va)&set(te))==0
    ox=np.load(art/'ood_predictions.npz')
    report={'design':{'mlp_widths':[208,208,180], 'seeds':[42,43,44],
        'primary_seed':42, 'epochs_max':150,'batch_size':256,'optimizer':'Adam 0.001',
        'input_scaling':SCALE.tolist(),'same_saved_splits':True,'counts':{'train':len(tr),'validation':len(va),'test':len(te)},
        'gnn_status':'fixed validation-selected checkpoint, seed 42'},
        'environment':{'platform':platform.platform(),'python':platform.python_version(),
                       'tensorflow':tf.__version__,'numpy':np.__version__,'intra_threads':4,'inter_threads':2},'mlp_runs':[]}
    primary=None
    for seed in [42,43,44]:
        tf.keras.utils.set_random_seed(seed)
        model=mlp_model(seed)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='mse')
        trds=tf.data.Dataset.from_tensor_slices((np.float32(x[tr]/SCALE),np.float32((y[tr]-mean)/std))).shuffle(24000,seed=42).batch(256).prefetch(2)
        vads=tf.data.Dataset.from_tensor_slices((np.float32(x[va]/SCALE),np.float32((y[va]-mean)/std))).batch(256).prefetch(2)
        path=out/f'mlp_seed_{seed}.weights.h5'
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(path),save_weights_only=True,save_best_only=True,monitor='val_loss'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=8,min_lr=1e-5),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=24,restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(str(out/f'mlp_seed_{seed}_history.csv'))]
        start=time.perf_counter()
        hist=model.fit(trds,validation_data=vads,epochs=150,callbacks=callbacks,verbose=2)
        fit_seconds=time.perf_counter()-start
        model.load_weights(str(path))
        result={'seed':seed,'parameters':model.count_params(),'fit_seconds':fit_seconds,
            'epochs_run':len(hist.history['loss']),'best_epoch':int(np.argmin(hist.history['val_loss'])+1)}
        for name,ix,iy in [('validation',x[va],y[va]),('test',x[te],y[te]),('ood',ox['x'],ox['truth'])]:
            pred=predict_mlp(model,ix,mean,std)
            result[name]=score(ix,iy,pred,std)
            np.savez_compressed(out/f'mlp_seed_{seed}_{name}.npz',prediction=pred)
        report['mlp_runs'].append(result)
        (out/'comparison.json').write_text(json.dumps(report,indent=2))
        print('FINISHED MLP',seed,result['test']['mean_r2'],flush=True)
        if seed==42: primary=model

    test=np.load(art/'test_predictions.npz')
    report['gnn']=score(test['x'],test['truth'],test['prediction'],std)
    report['gnn']['parameters']=86547
    report['gnn']['ood']=score(ox['x'],ox['truth'],ox['prediction'],std)
    # Validate optimized inference preprocessing against reference features.
    g=fast_graph(x[te[:256]])
    for i in range(8):
        n,e=features(x[te[i]])
        np.testing.assert_allclose(g.node_sets['bodies']['features'][i],n,atol=1e-6)
        np.testing.assert_allclose(g.edge_sets['interactions']['features'][i],e,atol=1e-6)
    gnn=build_model(g.spec); gnn.load_weights(str(art/'best.weights.h5'))
    check=gnn(g,training=False).numpy()*std+mean
    delta=float(np.max(np.abs(check-test['prediction'][:256])))
    np.testing.assert_allclose(check,test['prediction'][:256],atol=2e-4,rtol=1e-5)
    report['preprocessing_check_max_abs_difference']=delta
    np.testing.assert_allclose(solve(test['x']),test['truth'],atol=1e-12)

    # A separately shaped model input shares exactly the same architecture/weights.
    # tf.function traces are warmed before measurement. .numpy() synchronizes work.
    benchmark=[]
    raw=[]
    rng=np.random.default_rng(2026)
    for batch in [1,256,3000]:
        xb=x[te[:batch]]
        graph=fast_graph(xb)
        bm=build_model(graph.spec); bm.load_weights(str(art/'best.weights.h5'))
        gm=tf.function(lambda v:bm(v,training=False))
        mm=tf.function(lambda v:primary(v,training=False))
        mx=np.float32(xb/SCALE)
        def analytic(): return solve(xb)
        def mlp_end():
            xx=validate(xb)
            return mm(tf.convert_to_tensor(np.float32(xx/SCALE))).numpy()*std+mean
        def gnn_end(): return gm(fast_graph(xb)).numpy()*std+mean
        def mlp_core(): return mm(mx).numpy()
        def gnn_core(): return gm(graph).numpy()
        funcs={'analytical_end_to_end':analytic,'mlp_end_to_end':mlp_end,
               'gnn_end_to_end':gnn_end,'mlp_model_only':mlp_core,'gnn_model_only':gnn_core}
        for f in funcs.values():
            for _ in range(5): f()
        samples={k:[] for k in funcs}
        # Interleave methods in random order each round to reduce order bias.
        for repeat in range(60):
            for name in rng.permutation(list(funcs)):
                start=time.perf_counter_ns(); value=funcs[name](); ns=time.perf_counter_ns()-start
                assert np.isfinite(value).all()
                samples[name].append(ns/1e6)
                raw.append({'batch_size':batch,'repeat':repeat,'method':name,'ms':ns/1e6})
        for name,values in samples.items():
            benchmark.append({'batch_size':batch,'method':name,'repeats':len(values),
                'median_ms':float(np.median(values)),'p10_ms':float(np.percentile(values,10)),
                'p90_ms':float(np.percentile(values,90)),'microseconds_per_case':float(np.median(values)*1000/batch)})
        print('BENCHMARK FINISHED',batch,flush=True)
    report['timing']=benchmark
    report['timing_protocol']={'device':'CPU','clock':'perf_counter_ns','warmups':5,'repeats':60,
        'order':'randomized/interleaved','includes':'input validation, scaling/graph construction, tensor conversion, forward pass, NumPy output, target inverse scaling',
        'excludes':'imports, model loading, tracing, disk I/O, console printing; all models resident',
        'gnn_preprocessing':'optimized vectorized equivalent; no TFRecord roundtrip',
        'precision':'analytical float64, neural computations float32; targets restored with float64 scalers'}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    (out/'timing_raw.json').write_text(json.dumps(raw,indent=2))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',default='artifacts')
    p.add_argument('--out',default='comparison')
    main(p.parse_args())
