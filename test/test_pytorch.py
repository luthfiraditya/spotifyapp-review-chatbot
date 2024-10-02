import torch
print(torch.cuda.is_available())  # Should return True if a GPU is available
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.get_device_name(0))  # Prints the name of the first GPU
