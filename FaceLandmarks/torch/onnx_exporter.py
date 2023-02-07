import torch
import numpy
import onnx
import onnxruntime
import time

target_name = "./build/landmarks_net.onnx"

model = torch.load("./build/landmarks_net.pth").to(torch.device('cpu'))
model.eval()
dummy_input = torch.randn(1, 3, 100, 100)

torch.onnx.export(model,
                  dummy_input,
                  target_name,
                  verbose=False,
                  export_params=True,
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}}
                  )

onnx_model = onnx.load(target_name)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(target_name)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
numpy.testing.assert_allclose(to_numpy(model(dummy_input)), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

print("Performance benchmark:")
repeats = 30

t = 0
for i in range(repeats):
    x = torch.randn(1, 3, 100, 100)
    t0 = time.perf_counter()
    ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x)})
    t += time.perf_counter() - t0
t /= repeats
print(f"ONNX runtime: {(time.perf_counter() - t0)*1000:.1f} ms")

t = 0
for i in range(repeats):
    x = torch.randn(1, 3, 100, 100)
    t0 = time.perf_counter()
    model(x)
    t += time.perf_counter() - t0
t /= repeats
print(f"PyTorch runtime: {(time.perf_counter() - t0)*1000:.1f} ms")