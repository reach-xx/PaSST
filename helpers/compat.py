"""Compatibility shims for running this repo on newer Python.

sacred 0.8.5 does `ctypes.cdll.msvcrt` and only catches OSError. On Linux with
Python 3.12 that access raises AttributeError instead, so import the helper
before importing sacred.
"""

import ctypes


def _patch_sacred_msvcrt():
    orig = ctypes.LibraryLoader.__getattr__

    def __getattr__(self, name):
        try:
            return orig(self, name)
        except AttributeError:
            if name == "msvcrt":
                raise OSError(name)
            raise

    ctypes.LibraryLoader.__getattr__ = __getattr__


_patch_sacred_msvcrt()
