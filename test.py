import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 사용
    print("GPU 사용 가능:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")  # CPU 사용
    print("GPU 사용 불가능")

