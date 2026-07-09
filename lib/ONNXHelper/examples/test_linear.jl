using ONNXHelper

X = TensorValueInfoProto("X", Float32, (nothing, nothing))
A = TensorValueInfoProto("A", Float32, (nothing, nothing))
B = TensorValueInfoProto("B", Float32, (nothing, nothing))

Y = TensorValueInfoProto("Y", Float32, (nothing,))

node1 = NodeProto("MatMul", ["X", "A"], ["XA"])
node2 = NodeProto("Add", ["XA", "B"], ["Y"])

graph = GraphProto("main_graph", [node1, node2], [X, A, B], [Y])

model = ModelProto(graph)

savemodel("models/linear.onnx", model)
