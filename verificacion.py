import torch
print(torch.version.cuda)
print(torch.cuda.is_available())


print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Debería mostrar "GeForce GTX 1060 6GB"
