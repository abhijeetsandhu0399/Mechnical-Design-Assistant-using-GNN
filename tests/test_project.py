import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from mechanics import generate,solve,residuals
from graph_data import tf,tfgnn,graph,decode,write_records,dataset,SOURCE,TARGET
from model import build_model

class MechanicsTests(unittest.TestCase):
    def test_hand_calculated_case(self):
        x=np.array([[40,60,-30,.08,.16,.05,-.08]])
        expected=[-40,-60,30,3,-2,0,-40,-30,2.5,0,-30,27.5,3,-40,-60,30,4.8,-7.6,-4.8]
        np.testing.assert_allclose(solve(x)[0],expected,atol=1e-12)

    def test_all_body_balances(self):
        x,y=generate(10000)
        np.testing.assert_allclose(residuals(x,y),0,atol=1e-11)

    def test_independent_shaft_linear_system(self):
        x,y=generate(100)
        for xi,yi in zip(x,y):
            fx,fy,fz,d,L,r,gz=xi
            # Unknowns Bx,By,Bz,Cy,Cz,Tx; force + moment about B.
            mat=np.array([[1,0,0,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],
                          [0,0,0,0,0,1],[0,0,0,0,-L,0],[0,0,0,L,0,0]],float)
            rhs=-np.array([fx,fy,fz,-r*fy,r*fx-d*fz,d*fy])
            expected=np.linalg.solve(mat,rhs)
            np.testing.assert_allclose(yi[[6,7,8,10,11,12]],expected,atol=1e-12)

    def test_load_reversal_and_zero(self):
        x,y=generate(10); x[:,:3]*=-1
        np.testing.assert_allclose(solve(x),-y)
        x[:,:3]=0; np.testing.assert_array_equal(solve(x),0)

    def test_invalid_geometry(self):
        for x in [[1,2,3,.1,0,.05,-.08],[1,2,3,.2,.1,.05,-.08],[1,2,float('nan'),.1,.2,.05,-.08]]:
            with self.assertRaises(ValueError): solve(np.array([x]))

class GraphTests(unittest.TestCase):
    def test_record_roundtrip_and_label_removal(self):
        x,y=generate(5)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'graphs.tfrecord'; write_records(path,x,y)
            records=list(tf.data.TFRecordDataset(str(path)))
            self.assertEqual(len(records),5)
            for i,r in enumerate(records):
                ex=tf.train.Example.FromString(r.numpy())
                self.assertEqual(ex,tfgnn.write_example(graph(x[i],y[i])))
                self.assertIn('context/label',ex.features.feature)
                g,label=decode(r)
                self.assertNotIn('label',g.context.features)
                np.testing.assert_allclose(label.numpy(),y[i],rtol=1e-6,atol=1e-6)
                np.testing.assert_array_equal(g.edge_sets['interactions'].adjacency.source,SOURCE)
            # Missing required labels must cause a parsing error.
            ex.features.feature.pop('context/label')
            with self.assertRaises(tf.errors.InvalidArgumentError): decode(tf.constant(ex.SerializeToString()))

    def test_batch_isolation_permutation_and_weight_reload(self):
        x,y=generate(2)
        graphs=[graph(xi,yi).replace_features(context={}) for xi,yi in zip(x,y)]
        g=graphs[0]; e=g.edge_sets['interactions']; nodes=g.node_sets['bodies']['features'].numpy()
        order=np.array([2,0,3,1]); inverse=np.argsort(order)
        permuted=tfgnn.GraphTensor.from_pieces(context=g.context,
            node_sets={'bodies':tfgnn.NodeSet.from_fields(sizes=[4],features={'features':nodes[order]})},
            edge_sets={'interactions':tfgnn.EdgeSet.from_fields(sizes=[12],features=e.features,
                adjacency=tfgnn.Adjacency.from_indices(source=('bodies',inverse[SOURCE].astype('int32')),target=('bodies',inverse[TARGET].astype('int32'))))})
        def batch(gs):
            encoded=[tfgnn.write_example(v).SerializeToString() for v in gs]
            return next(iter(tf.data.Dataset.from_tensor_slices(encoded).map(lambda s:tfgnn.parse_single_example(g.spec,s)).batch(len(gs))))
        b=batch(graphs); model=build_model(b.spec)
        expected=model(b).numpy()
        np.testing.assert_allclose(model(batch([graphs[0]])).numpy(),expected[:1],atol=1e-5)
        np.testing.assert_allclose(model(batch([permuted])).numpy(),expected[:1],atol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            file=str(Path(tmp)/'model.weights.h5'); model.save_weights(file)
            restored=build_model(b.spec); restored.load_weights(file)
            np.testing.assert_allclose(restored(b).numpy(),expected,atol=1e-6)

if __name__=='__main__': unittest.main()
