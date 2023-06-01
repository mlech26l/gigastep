from torch.utils.dlpack import from_dlpack, to_dlpack
import jax.dlpack

import os

if (
    "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    and "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ
):
    print(
        "WARNING: XLA_PYTHON_CLIENT_PREALLOCATE not set, this may cause out-of-memory errors!"
    )


def jax2torch(x):
    x = jax.dlpack.to_dlpack(x, take_ownership=False)
    x = from_dlpack(x)
    return x


def torch2jax(x):
    x = to_dlpack(x)
    x = jax.dlpack.from_dlpack(x)
    return x