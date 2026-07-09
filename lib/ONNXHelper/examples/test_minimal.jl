using ONNXHelper

tensor = TensorProto([1.0f0])

node = NodeProto("Constant", [], ["y"], (value=tensor,))
y = TensorValueInfoProto("y", Float32, 1)

graph = GraphProto("main_graph", [node], [], [y])

model = ModelProto(graph)

savemodel("models/minimal.onnx", model)
