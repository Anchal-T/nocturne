"""Process-wide compatibility hooks for third-party worker processes."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _apply_ray_compatibility() -> None:
    """Load the hook without importing nocturne's native extension package."""
    module_path = Path(__file__).parent / "nocturne/utils/ray_compat.py"
    spec = spec_from_file_location("_nocturne_ray_compat", module_path)
    if spec is None or spec.loader is None:
        return
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        # This hook is optional for processes without the project dependencies.
        return
    module.patch_legacy_gym_monitor()


_apply_ray_compatibility()
