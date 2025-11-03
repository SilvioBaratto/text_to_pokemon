"""Memory cleanup utilities for training."""

import gc
import signal
import atexit
import sys
import torch


# Global references for cleanup
_model_ref = None
_optimizer_ref = None
_device_ref = None
_cleanup_done = False


def cleanup_memory():
    """Free GPU/CPU memory when training stops."""
    global _cleanup_done

    if _cleanup_done:
        return  # Prevent double cleanup

    _cleanup_done = True

    print("\n" + "="*80)
    print("CLEANING UP MEMORY")
    print("="*80)

    # Delete model and optimizer to free memory
    global _model_ref, _optimizer_ref, _device_ref

    if _model_ref is not None:
        del _model_ref
        print("✓ Deleted model")

    if _optimizer_ref is not None:
        del _optimizer_ref
        print("✓ Deleted optimizer")

    # Clear GPU cache based on device type
    if _device_ref is not None:
        if _device_ref.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Cleared CUDA cache")

            # Print memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"  GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        elif _device_ref.type == 'mps':
            torch.mps.empty_cache()
            print("✓ Cleared MPS (Apple Silicon) cache")

    # Force garbage collection
    gc.collect()
    print("✓ Ran garbage collection")

    print("="*80)
    print("Memory cleanup complete!")
    print("="*80 + "\n")


def signal_handler(_signum, _frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Received interrupt signal (Ctrl+C)")
    print("Stopping training and cleaning up memory...")
    cleanup_memory()
    sys.exit(0)


def register_cleanup_handlers():
    """Register cleanup handlers for Ctrl+C and normal exit."""
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    atexit.register(cleanup_memory)  # Normal exit


def set_cleanup_references(model, optimizer, device):
    """Set global references for cleanup.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        device: torch.device
    """
    global _model_ref, _optimizer_ref, _device_ref
    _model_ref = model
    _optimizer_ref = optimizer
    _device_ref = device
