import torch
import numpy
import onnx
import onnxruntime
import time
import neuralnet

# ------------------------------------------------------------------------------------
filters = 16
size = (100, 100)
layers = [1, 1, 2, 1]
model_name = f"resnet{sum(layers) * 2 + 2}_{filters}f_{size[0]}@200bbox"
# ------------------------------------------------------------------------------------

model = torch.load(f"./weights/{model_name}_test.pth").to(torch.device('cpu'))
model.eval()

# Save model's state_dict
target_name_pytorch = f"./runs/{model_name}.pth"
torch.save(model.state_dict(), target_name_pytorch)
model2 = neuralnet.ResNet(neuralnet.BasicBlock, filters, layers)
model2.load_state_dict(torch.load(target_name_pytorch))
model2.eval()

dummy_input = torch.randn(1, 3, 100, 100)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


numpy.testing.assert_allclose(to_numpy(model(dummy_input)[0]), to_numpy(model2(dummy_input)[0]), rtol=1e-03, atol=1e-05)
numpy.testing.assert_allclose(to_numpy(model(dummy_input)[1]), to_numpy(model2(dummy_input)[1]), rtol=1e-03, atol=1e-05)
print("Exported model has been tested with load_state_dict, and the result looks good!\n")

# ONNX
target_name_onnx = f"./runs/{model_name}.onnx"
torch.onnx.export(model,
                  dummy_input,
                  target_name_onnx,
                  verbose=False,
                  export_params=True,
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['angles', 'landmarks'],
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'angles': {0: 'batch_size'},
                                'landmarks': {0: 'batch_size'}}
                  )

onnx_model = onnx.load(target_name_onnx)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(target_name_onnx)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results

t_angles, t_landmarks = model(dummy_input)
numpy.testing.assert_allclose(to_numpy(t_angles), ort_outs[0], rtol=1e-03, atol=1e-05)
numpy.testing.assert_allclose(to_numpy(t_landmarks), ort_outs[1], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

print("Performance benchmark:")
repeats = 30
warmup = 5

t = 0
for i in range(repeats+warmup):
    x = torch.randn(1, 3, 100, 100)
    t0 = time.perf_counter()
    ort_session.run(None, {ort_session.get_inputs()[0].name: to_numpy(x)})
    if i > warmup:
        t += time.perf_counter() - t0
print(f" - ONNX runtime: {t/repeats*1000:.1f} ms")

t = 0
for i in range(repeats+warmup):
    x = torch.randn(1, 3, 100, 100)
    t0 = time.perf_counter()
    model(x)
    if i > warmup:
        t += time.perf_counter() - t0
print(f" - PyTorch runtime: {t/repeats*1000:.1f} ms")