using ONNXHelper

X = TensorValueInfoProto("X", Float32, (nothing, nothing))
A = TensorProto([0.5f0, -0.6f0]; name="A")
C = TensorProto([0.4f0]; name="C")

Y = TensorValueInfoProto("Y", Float32, (nothing,))

node1 = NodeProto("MatMul", ["X", "A"], ["XA"])
node2 = NodeProto("Add", ["XA", "C"], ["Y"])

graph = GraphProto("main_graph", [node1, node2], [X], [Y], [A, C])

model = ModelProto(graph)

savemodel("models/linear.onnx", model)
