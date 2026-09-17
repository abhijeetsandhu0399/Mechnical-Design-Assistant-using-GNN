"""Predict all 19 mechanical quantities from applied force and geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from graph_data import tf,tfgnn,graph
from model import build_model
from mechanics import NAMES,UNITS,solve,residuals,validate

def predict(x,artifacts='artifacts'):
    x=validate(np.asarray(x,dtype=float))
    folder=Path(artifacts); norm=np.load(folder/'normalization.npz')
    # Labels here are placeholders removed before constructing model input.
    serialized=[tfgnn.write_example(graph(v,np.zeros(19))).SerializeToString() for v in x]
    from graph_data import decode
    ds=tf.data.Dataset.from_tensor_slices(serialized).map(decode).batch(256)
    model=build_model(ds.element_spec[0]); model.load_weights(str(folder/'best.weights.h5'))
    return model.predict(ds,verbose=0)*norm['std']+norm['mean']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force',type=float,nargs=3,default=[40,60,-30],metavar=('FX','FY','FZ'))
    p.add_argument('--geometry',type=float,nargs=4,default=[.08,.16,.05,-.08],metavar=('D','L','R','GZ'))
    p.add_argument('--artifacts',default='artifacts')
    a=p.parse_args(); x=np.array([a.force+a.geometry]); y=predict(x,a.artifacts)[0]; exact=solve(x)[0]
    print(json.dumps({n:{'predicted':float(v),'analytical':float(t),'unit':u} for n,u,v,t in zip(NAMES,UNITS,y,exact)},indent=2))
