import tensorrt as trt

TRT_LOGGER = trt.Logger()
engine_file_path = "License.engine"

with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

    print("Inputs:")
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            print(f"  Name: {engine.get_binding_name(i)}")
            print(f"  Shape: {engine.get_binding_shape(i)}")
            print(f"  Dtype: {engine.get_binding_dtype(i)}")

    print("Outputs:")
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            print(f"  Name: {engine.get_binding_name(i)}")
            print(f"  Shape: {engine.get_binding_shape(i)}")
            print(f"  Dtype: {engine.get_binding_dtype(i)}")
