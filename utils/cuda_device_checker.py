import torch

def print_available_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}")
        for device_id in range(num_devices):
            print(f"Device ID: {device_id}")
            print(f"Device Name: {torch.cuda.get_device_name(device_id)}")
            print(f"Device Memory Allocated: {torch.cuda.memory_allocated(device_id)} bytes")
            print(f"Device Memory Cached: {torch.cuda.memory_reserved(device_id)} bytes")
            print("-" * 40)
    else:
        print("CUDA is not available on this system.")

print_available_cuda_devices()