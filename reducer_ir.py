import onnx

# Load the ONNX model
model = onnx.load("yolo11n.onnx")

# Modify the IR version (e.g., change to 9 if your runtime supports only up to 9)
model.ir_version = 9

# Save the modified model
onnx.save(model, "yolo11n_reduced.onnx")

print(f"Modified IR version: {model.ir_version}")
