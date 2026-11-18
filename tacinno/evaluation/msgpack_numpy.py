# Copyright 2024 Physical Intelligence. Licensed under the Apache License, Version 2.0.
# Vendored from openpi-client (openpi_client/msgpack_numpy.py) without functional changes, so that the UniVTAC client
# runs inside the simulator environment without installing openpi. The encoding is byte-compatible with openpi_client.
"""NumPy array support for msgpack (the wire format of the TacInno-VLA policy server).

Arrays are sent as ``{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": tuple}`` and NumPy scalars as
``{b"__npgeneric__": True, b"data": item, b"dtype": str}``. Object arrays are rejected, so no pickle is ever involved.
"""
import functools

import msgpack
import numpy as np

__all__ = ["pack_array", "unpack_array", "Packer", "packb", "Unpacker", "unpackb"]


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
