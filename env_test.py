import torch
import tensorflow as tf

print("--- PyTorch Verification ---")
print(f"PyTorch Version: {torch.__version__}")
is_pytorch_cuda_available = torch.cuda.is_available()
print(f"PyTorch CUDA Available: {is_pytorch_cuda_available}")
if is_pytorch_cuda_available:
    print(f"  - Device Name: {torch.cuda.get_device_name(0)}")

print("\n--- TensorFlow Verification ---")
try:
    print(f"TensorFlow Version: {tf.__version__}")
    is_tf_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    print(f"TensorFlow GPU Available: {is_tf_gpu_available}")
except Exception as e:
    print(f"An error occurred during TensorFlow check: {e}")