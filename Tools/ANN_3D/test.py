'''
import torch

print(f"Is CUDA supported by this system?  {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()}")
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
'''



'''
import torch

# Creating a test tensor
x = torch.randint(1, 100, (100, 100))

# Checking the device name:
# Should return 'cpu' by default
print(x.device)

# Applying tensor operation
res_cpu = x ** 2

# Transferring tensor to GPU
x = x.to(torch.device('cuda'))

# Checking the device name:
# Should return 'cuda:0'
print(x.device)

# Applying same tensor operation
res_gpu = x ** 2

# Checking the equality
# of the two results
assert torch.equal(res_cpu, res_gpu.cpu())
'''




import torch
import torchvision.models as models

# Making the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiating a pre-trained model
model = models.resnet18(pretrained=True)

# Transferring the model to a CUDA enabled GPU
model = model.to(device)

# Now the reader can continue the rest of the workflow
# including training, cross validation, etc!
