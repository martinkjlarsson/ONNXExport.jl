using ONNXHelper

zero = TensorProto([0]; name="zero")

X = TensorValueInfoProto("X", Float32, (nothing, nothing))
Y = TensorValueInfoProto("Y", Float32, (nothing,)) # TODO: Allow scalar nothing/integer for 1D vectors?

rsum = NodeProto("ReduceSum", ["X"], ["rsum"])
cond = NodeProto("Greater", ["rsum", "zero"], ["cond"])

then_out = TensorValueInfoProto("then_out", Float32, (nothing,))
then_cst = TensorProto([1.0f0])

then_const_node = NodeProto("Constant", [], ["then_out"], (value=then_cst,); name="cst1")

then_body = GraphProto("then_body", [then_const_node], [], [then_out])

else_out = TensorValueInfoProto("else_out", Float32, (5,))
else_cst = TensorProto([-1.0f0])

else_const_node = NodeProto("Constant", [], ["else_out"], (value=else_cst,); name="cst2")

else_body = GraphProto("else_body", [else_const_node], [], [else_out])

if_node = NodeProto("If", ["cond"], ["Y"], (then_branch=then_body, else_branch=else_body))

graph = GraphProto("main_graph", [rsum, cond, if_node], [X], [Y], [zero])
model = ModelProto(graph)

savemodel("models/if.onnx", model)
