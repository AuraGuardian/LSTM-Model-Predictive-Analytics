import os
import sys

def setup_gpu():
    # Add CUDA to the system path
    cuda_path = os.path.join(os.environ.get('CUDA_PATH', ''), 'bin')
    cudnn_path = os.path.join(os.path.dirname(os.path.dirname(os.__file__)), 'Lib', 'site-packages', 'nvidia', 'cudnn', 'lib')
    
    # Add paths to environment
    if cuda_path not in os.environ['PATH']:
        os.environ['PATH'] = f"{cuda_path};{os.environ['PATH']}"
    if cudnn_path not in os.environ['PATH']:
        os.environ['PATH'] = f"{cudnn_path};{os.environ['PATH']}"
    
    # Set other required environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
    
    # Verify CUDA is available
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print("GPU available:", tf.config.list_physical_devices('GPU'))
        print("CUDA available:", tf.test.is_built_with_cuda())
        print("cuDNN version:", tf.sysconfig.get_build_info().get('cudnn_version', 'Not found'))
        return True
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        return False

if __name__ == "__main__":
    if setup_gpu():
        print("GPU setup completed successfully!")
    else:
        print("Failed to set up GPU. Running in CPU mode.")
        sys.exit(1)
