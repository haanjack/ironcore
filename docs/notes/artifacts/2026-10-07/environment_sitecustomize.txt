# Validation-only workaround for this host's native-library import-order crash.
import triton
import sys
import types
# TensorBoard supports a lightweight stub when TensorFlow is unavailable.
sys.modules['tensorboard.compat.notf'] = types.ModuleType('tensorboard.compat.notf')
