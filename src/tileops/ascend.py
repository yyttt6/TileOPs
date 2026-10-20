"""Load Ascend 950 implementations through the standard backend entry point.

Registration adds NPU implementations to the existing kernel interfaces. Device
selection keeps CUDA implementations available in the same installation. No
kernel is built and no device property is read during registration.
"""

import importlib
import importlib.util

# A CUDA installation does not require torch_npu. Import failures with the NPU
# runtime installed propagate to the backend registry's diagnostic channel.
if importlib.util.find_spec("torch_npu") is not None:
    importlib.import_module("tileops.kernels.ascend")
