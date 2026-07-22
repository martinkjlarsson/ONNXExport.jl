using ONNXHelper

new_domain = "custom"
opset_imports = [OperatorSetIdProto("", 14), OperatorSetIdProto(new_domain, 1)]

att = AttributeProto("name", "bias", TensorProto)
cst = NodeProto("Constant", [], ["B"], [att])
node1 = NodeProto("MatMul", ["X", "A"], ["XA"])
node2 = NodeProto("Add", ["XA", "B"], ["Y"])

linear_regression = FunctionProto(
    new_domain,             # domain name
    "LinearRegression",     # function name
    ["X", "A"],             # input names
    ["Y"],                  # output names
    [cst, node1, node2],    # nodes
    ["bias"];               # attribute names
    opset_import=opset_imports,
)

# Let's use it in a graph.
X = TensorValueInfoProto("X", Float32, (nothing, nothing))
A = TensorValueInfoProto("A", Float32, (nothing, nothing))
B = TensorValueInfoProto("B", Float32, (nothing, nothing))
Y = TensorValueInfoProto("Y", Float32, (nothing,))

graph = GraphProto(
    "main_graph",
    [
        NodeProto(
            "LinearRegression",
            ["X", "A"],
            ["Y1"],
            # bias is now an argument of the function and is defined as a tensor
            (bias=TensorProto([0.67f0]; name="former_B"),);
            domain=new_domain,
        ),
        NodeProto("Abs", ["Y1"], ["Y"]),
    ],
    [X, A],
    [Y],
)

model = ModelProto(graph; opset_import=opset_imports, functions=[linear_regression])

save_model("models/function.onnx", model)
