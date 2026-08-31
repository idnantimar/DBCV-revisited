from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [

    # NOTE: apcd calculation requires 1/0, 1/inf semantics; strictly not suitable for `-ffinite-math-only`.
    Extension(
        "DBCV.utils.dsc._apcd_mrd",
        ["DBCV/utils/dsc/_apcd_mrd.pyx"],
        extra_compile_args=["-O3"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),

    # NOTE: mst calculation is dominated by if-else; `-ffinite-math-only` not much beneficial here.
    Extension(
        "DBCV.utils.dsc._mst",
        ["DBCV/utils/dsc/_mst.pyx"],
        extra_compile_args=["-O3"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),

    # NOTE: runtime contribution of dspc is <5% of total.
    Extension(
        "DBCV.utils.dspc._ckdtree",
        sources=[
            "DBCV/utils/dspc/_ckdtree.pyx",
            "DBCV/utils/dspc/ckdtree/src/build.cxx",
            "DBCV/utils/dspc/ckdtree/src/query.cxx",
            "DBCV/utils/dspc/ckdtree/src/query_ball_point.cxx",
        ],
        include_dirs=["DBCV/utils/dspc/ckdtree/src"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),

]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "initializedcheck": False,
            "cdivision": True,
            "cpow": True,
        },
    ),
    include_dirs=[np.get_include()],
)
