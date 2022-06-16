import torch

INPUT_DICT = './weight/best.pth'
OUT_ONNX = './weight/best.onnx'

x = torch.randn(1, 3, 224, 224)
input_names = ["input"]
out_names = ["output"]

model= torch.load(INPUT_DICT, map_location=torch.device('cpu'))
model.eval()

torch.onnx._export(model, x, OUT_ONNX, export_params=True, training=False, input_names=input_names, output_names=out_names)
print('please run: python -m onnxsim test.onnx test_sim.onnx\n')
