using ONNXHelper

X = TensorValueInfoProto("X", Float32, (nothing, nothing))
A = TensorValueInfoProto("A", Float32, (nothing, nothing))
B = TensorValueInfoProto("B", Float32, (nothing, nothing))
Y = TensorValueInfoProto("Y", Float32, (nothing,))

node_transpose = NodeProto("Transpose", ["A"], ["tA"], (perm=[1, 0],))

node1 = NodeProto("MatMul", ["X", "tA"], ["XA"])
node2 = NodeProto("Add", ["XA", "B"], ["Y"])

graph = GraphProto("main_graph", [node_transpose, node1, node2], [X, A, B], [Y])

model = ModelProto(graph)

save_model("models/attribute.onnx", model)
