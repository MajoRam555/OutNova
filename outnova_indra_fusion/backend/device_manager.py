import logging

logger = logging.getLogger(__name__)

_device: str | None = None


def get_device() -> str:
    """Returns 'cuda' if available, else 'cpu'. Result cached after first call."""
    global _device
    if _device is not None:
        return _device
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    return _device


def log_device_info() -> None:
    """Logs GPU/CPU info on startup."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        logger.info(
            f"[DEVICE] torch={torch.__version__}  cuda_build={torch.version.cuda}"
            f"  cuda_available={cuda_ok}  device_count={torch.cuda.device_count()}"
        )
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            logger.info(f"[DEVICE] cuda:0 → {name}  VRAM_total={vram} MB")
        else:
            logger.info("[DEVICE] Sin CUDA disponible — usando CPU.")
    except Exception as e:
        logger.warning(f"[DEVICE] No se pudo leer info de GPU: {e}")


def get_gpu_status() -> dict:
    """Returns GPU/CPU status dict for the /gpu/status endpoint."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        info = {
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": cuda_ok,
            "device_count": torch.cuda.device_count(),
            "active_device": get_device(),
            "gpu_name": None,
            "vram_total_mb": None,
            "vram_free_mb": None,
            "vram_used_mb": None,
        }
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            free, _ = torch.cuda.mem_get_info(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_mb"] = round(props.total_memory / (1024 ** 2))
            info["vram_free_mb"] = round(free / (1024 ** 2))
            info["vram_used_mb"] = round((props.total_memory - free) / (1024 ** 2))
        return info
    except Exception as e:
        return {"error": str(e), "active_device": "cpu"}
