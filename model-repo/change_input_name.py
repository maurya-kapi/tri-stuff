import onnx

# Load model
model = onnx.load("PaddleOCR-Detection/1/model.onnx")

# Rename input
old_input = "x"
new_input = "inp"

for input in model.graph.input:
    if input.name == old_input:
        input.name = new_input

for node in model.graph.node:
    node.input[:] = [new_input if i == old_input else i for i in node.input]

for init in model.graph.initializer:
    if init.name == old_input:
        init.name = new_input

# Rename output
old_output = "fetch_name_0"
new_output = "fetch_name"

for output in model.graph.output:
    if output.name == old_output:
        output.name = new_output

for node in model.graph.node:
    node.output[:] = [new_output if o == old_output else o for o in node.output]

# Save new model
onnx.save(model, "renamed_io_model.onnx")
print("✅ Input and output renamed successfully.")
