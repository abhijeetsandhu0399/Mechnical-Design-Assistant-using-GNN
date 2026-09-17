"""Generate, serialize, train, and evaluate the mechanical GNN end to end."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from graph_data import tf,tfgnn,write_records,dataset,graph
from model import build_model
from mechanics import generate,residuals,NAMES,UNITS,INPUTS

def metrics(y,p,scale):
    err=p-y; var=np.sum((y-y.mean(0))**2,0)
    scores=[]
    for i,name in enumerate(NAMES):
        scores.append(dict(output=name,unit=UNITS[i],mae=float(np.abs(err[:,i]).mean()),
                           rmse=float(np.sqrt(np.mean(err[:,i]**2))),
                           r2=float(1-np.sum(err[:,i]**2)/var[i]) if var[i]>1e-12 else None))
    return dict(per_output=scores,mean_r2=float(np.mean([v['r2'] for v in scores if v['r2'] is not None])),
                standardized_rmse=float(np.sqrt(np.mean((err/scale)**2))))

def residual_metrics(x,p):
    r=residuals(x,p)
    return {name:{'force_rms_N':float(np.sqrt(np.mean(r[:,i,:3]**2))),
                  'moment_rms_Nm':float(np.sqrt(np.mean(r[:,i,3:]**2))),
                  'force_max_abs_N':float(np.max(np.abs(r[:,i,:3]))),
                  'moment_max_abs_Nm':float(np.max(np.abs(r[:,i,3:])))}
            for i,name in enumerate(['tooth_wheel','shaft','support'])}

def run(args):
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    start=time.time(); x,y=generate(args.samples,args.seed)
    ids=np.random.default_rng(args.seed+1).permutation(len(x))
    splits=dict(zip(['train','validation','test'],np.split(ids,[int(.8*len(x)),int(.9*len(x))])))
    mean=y[splits['train']].mean(0); std=y[splits['train']].std(0)
    std=np.where(std<1e-8,1.,std)
    np.savez(out/'normalization.npz',mean=mean,std=std)
    np.savez_compressed(out/'dataset.npz',x=x,y=y,**splits)
    # Independent residual implementation catches generator sign mistakes.
    assert np.max(np.abs(residuals(x,y)))<1e-10
    for name,idx in splits.items():
        print(f'Writing {name}: {len(idx)} complete graphs',flush=True)
        write_records(out/f'{name}.tfrecord',x[idx],((y[idx]-mean)/std).astype('float32'))
    from google.protobuf import text_format
    (out/'graph_schema.pbtxt').write_text(text_format.MessageToString(tfgnn.create_schema_pb_from_graph_spec(graph(x[0],y[0]).spec)))
    tr=dataset(out/'train.tfrecord',args.batch,True)
    va=dataset(out/'validation.tfrecord',args.batch)
    te=dataset(out/'test.tfrecord',args.batch)
    model=build_model(tr.element_spec[0])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='mse')
    (out/'model_summary.txt').write_text('')
    with (out/'model_summary.txt').open('w') as f:
        model.summary(print_fn=lambda s:f.write(s+'\n'))
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out/'best.weights.h5'),monitor='val_loss',save_best_only=True,save_weights_only=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=8,min_lr=1e-5),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=24,restore_best_weights=True),
               tf.keras.callbacks.CSVLogger(str(out/'history.csv'))]
    history=model.fit(tr,validation_data=va,epochs=args.epochs,callbacks=callbacks,verbose=2)
    model.load_weights(str(out/'best.weights.h5'))
    report={'config':vars(args),'versions':{'tensorflow':tf.__version__,'tensorflow_gnn':tfgnn.__version__,'numpy':np.__version__},
            'counts':{k:len(v) for k,v in splits.items()},'parameters':model.count_params(),
            'epochs_run':len(history.history['loss']),'best_epoch':int(np.argmin(history.history['val_loss'])+1)}
    for name,ds in [('validation',va),('test',te)]:
        idx=splits[name]; pred=model.predict(ds,verbose=0)*std+mean
        report[name]=metrics(y[idx],pred,std)
        report[name]['equilibrium']=residual_metrics(x[idx],pred)
        np.savez_compressed(out/f'{name}_predictions.npz',x=x[idx],truth=y[idx],prediction=pred,ids=idx)
    idx=splits['test']
    report['mean_baseline']=metrics(y[idx],np.broadcast_to(mean,y[idx].shape),std)
    # Affine baseline: fitted on training only, no interactions supplied.
    z=np.column_stack([np.ones(len(x)),x]); beta=np.linalg.lstsq(z[splits['train']],y[splits['train']],rcond=None)[0]
    report['linear_baseline']=metrics(y[idx],z[idx]@beta,std)
    report['analytic_equilibrium']=residual_metrics(x[idx],y[idx])
    ox,oy=generate(1000,args.seed+100,ood=True)
    write_records(out/'ood.tfrecord',ox,((oy-mean)/std).astype('float32'))
    op=model.predict(dataset(out/'ood.tfrecord',args.batch),verbose=0)*std+mean
    report['out_of_distribution']=metrics(oy,op,std)
    report['out_of_distribution']['equilibrium']=residual_metrics(ox,op)
    np.savez_compressed(out/'ood_predictions.npz',x=ox,truth=oy,prediction=op)
    report['elapsed_seconds']=time.time()-start
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k]['mean_r2'] for k in ['test','linear_baseline','out_of_distribution']},indent=2),flush=True)
    from report_results import make_report
    make_report(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=30000)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',default='artifacts')
    args=p.parse_args()
    if args.samples<100: p.error('Use at least 100 samples')
    run(args)
