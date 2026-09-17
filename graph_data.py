"""One complete GraphTensor per TFRecord; labels never enter the model."""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
from mechanics import points

# Nodes: tooth wheel (1), shaft (2), support (3), environment (E).
# Parallel bearing edges preserve distinct B and C connections.
SOURCE=np.array([3,0, 1,0, 2,1, 2,1, 3,1, 3,2],np.int32)
TARGET=np.array([0,3, 0,1, 1,2, 1,2, 1,3, 2,3],np.int32)

def features(x):
    a,d,c,g=points(x[None,:]); a,d,c,g=a[0],d[0],c[0],g[0]
    nodes=np.column_stack([np.eye(4),np.stack([d,c/2,g,np.zeros(3)])/.1]).astype('float32')
    edges=[]
    for k,p in enumerate([a,d,np.zeros(3),c,c,g]):
        for sign in (1.,-1.):
            edges.append(np.r_[p/.1,np.eye(6)[k],x[:3]/100*sign if k==0 else np.zeros(3),float(k==0),sign])
    return nodes,np.array(edges,np.float32)

def graph(x,y):
    nodes,edges=features(x)
    return tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(features={'label':tf.constant(y[None,:],tf.float32)}),
        node_sets={'bodies':tfgnn.NodeSet.from_fields(sizes=[4],features={'features':nodes})},
        edge_sets={'interactions':tfgnn.EdgeSet.from_fields(sizes=[12],features={'features':edges},
            adjacency=tfgnn.Adjacency.from_indices(source=('bodies',SOURCE),target=('bodies',TARGET)))})

def spec():
    return graph(np.array([1.,2.,3.,.08,.16,.05,-.08]),np.zeros(19)).spec

def write_records(path,x,y):
    if len(x) != len(y) or np.shape(y) != (len(x),19):
        raise ValueError('Expected one 19-component label per input graph')
    # Topology is fixed. Let the official serializer establish every structural
    # field, then replace only the varying arrays. This avoids rebuilding tens
    # of thousands of eager GraphTensors. Tests compare against the full writer.
    with tf.io.TFRecordWriter(str(path)) as writer:
        if not len(x):
            return
        example=tfgnn.write_example(graph(x[0],y[0]))
        for xi,yi in zip(x,y):
            nodes,edges=features(xi)
            for name,values in [('nodes/bodies.features',nodes),
                                ('edges/interactions.features',edges),
                                ('context/label',np.asarray(yi,np.float32))]:
                example.features.feature[name].float_list.value[:]=values.reshape(-1)
            writer.write(example.SerializeToString())

def decode(serialized):
    g=tfgnn.parse_single_example(spec(),serialized)
    label=g.context['label'][0]
    return g.replace_features(context={}),label

def dataset(path,batch=256,training=False):
    ds=tf.data.TFRecordDataset(str(path)).map(decode,num_parallel_calls=2).cache()
    if training: ds=ds.shuffle(24000,seed=42,reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(2)
