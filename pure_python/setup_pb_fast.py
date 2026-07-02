from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, setup


root = Path(__file__).resolve().parent

setup(
    name="pb_fast",
    ext_modules=[
        Extension(
            "pure_python._pb_fast",
            sources=[str(root / "_pb_fast.c")],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
)
