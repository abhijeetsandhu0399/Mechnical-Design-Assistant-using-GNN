"""Trainable edge-conditioned message passing, followed by graph regression."""
from graph_data import tf,tfgnn

def build_model(input_spec,width=64,steps=3):
    # Explicit integer seeds also avoid tf-keras 2.16's Python 3.12 randint bug.
    counter=iter(range(42,1000))
    def dense(units,activation=None):
        return tf.keras.layers.Dense(units,activation=activation,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=next(counter)))
    inp=tf.keras.layers.Input(type_spec=input_spec)
    graph=inp.merge_batch_to_components()
    graph=tfgnn.keras.layers.MapFeatures(
        node_sets_fn=lambda n,**kw:dense(width,'swish')(n['features']),
        edge_sets_fn=lambda e,**kw:dense(width,'swish')(e['features']))(graph)
    for _ in range(steps):
        graph=tfgnn.keras.layers.GraphUpdate(node_sets={'bodies':tfgnn.keras.layers.NodeSetUpdate(
            {'interactions':tfgnn.keras.layers.SimpleConv(
                message_fn=tf.keras.Sequential([dense(width,'swish'),dense(width,'swish')]),
                reduce_type='sum',receiver_tag=tfgnn.TARGET,sender_edge_feature=tfgnn.HIDDEN_STATE)},
            tfgnn.keras.layers.NextStateFromConcat(dense(width,'swish')))})(graph)
    h=tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'sum',node_set_name='bodies')(graph)
    h=dense(128,'swish')(h)
    return tf.keras.Model(inp,dense(19)(h))
