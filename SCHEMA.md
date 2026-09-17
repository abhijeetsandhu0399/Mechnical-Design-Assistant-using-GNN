# Mechanical graph and data schema

One example is one assembly, with one graph component. Every graph has node set `bodies` of size 4 and edge set `interactions` of size 12. Geometry and forces vary; topology is fixed. Two opposite directed edges encode each physical interaction, but unknown reaction magnitudes are **not** input edge features.

## Nodes

| Index | Identity | Reference point |
|---:|---|---|
| 0 | Tooth wheel, body 1 | D |
| 1 | Shaft, body 2 | C/2 |
| 2 | Support, body 3 | G |
| 3 | Environment E | B/origin |

`bodies/features`: float32 `[4,7]`: four identity one-hot entries plus three reference coordinates divided by 0.1 m. Reference points are encoding choices, not mass centroids. Node identity is carried by features, so reordering nodes and remapping endpoints preserves predictions (tested).

## Edges

| Edge rows | Type | Forward source -> target | Point |
|---|---|---|---|
| 0,1 | A applied force | E -> 1 | A |
| 2,3 | D tooth wheel/shaft | 2 -> 1 | D |
| 4,5 | B locating bearing | 3 -> 2 | B |
| 6,7 | C floating bearing | 3 -> 2 | C |
| 8,9 | External drive/brake torque | E -> 2 | C |
| 10,11 | G environment/support | E -> 3 | G |

The first row is forward; the next is reverse. B and C are distinct parallel graph connections. `interactions/features`: float32 `[12,14]` = location xyz / 0.1 m, six-way interaction one-hot, known force xyz / 100 N, known-force flag, direction (+1/-1). Only A has known applied force; its reverse has the opposite force. All unknown forces/couples remain zero placeholders distinguished by the known-force flag and edge type. This is a relational graph with a graph-level regression readout, not a learned reaction stored in each edge.

## Context and labels

Serialized `context/label`: float32 `[1,19]`, standardized using training-only statistics. `decode()` extracts its component row to `[19]` and replaces context features with `{}`. Batched labels are `[batch,19]`; batched graphs are merged into disjoint components in the model. No edges cross components.

Output order: `F_D12X,Y,Z`, `M_D12X,Y,Z`, `F_B23X,Y,Z`, `F_C23X,Y,Z`, `M_C2EX`, `F_G3EX,Y,Z`, `M_G3EX,Y,Z`. Force and couple directions follow the receiving-body conventions defined in the README.

`artifacts/graph_schema.pbtxt` is the machine-readable TF-GNN schema. `tfgnn.write_example()` manages feature names including `context/label`, `nodes/bodies.*` and `edges/interactions.*`. Each serialized example contains the context, graph sizes, node features, edge features, and endpoint indices. See the official [encoding and parsing guide](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md).
