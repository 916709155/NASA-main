import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
(torch.backends.cudnn.version())
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())