import os
from pathlib import Path

from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops = Path("pcdet/ops")
extensions = []
for source_dir in sorted(ops.glob("**/src")):
    module = source_dir.parent
    files = sorted(str(p) for p in source_dir.iterdir() if p.suffix in {".cpp", ".cu"})
    extensions.append(CUDAExtension(
        name=".".join((*module.parts, module.name + "_cuda")),
        sources=files,
        include_dirs=[str(Path(os.environ["CUDA_HOME"]) / "include")],
        extra_compile_args={"cxx": ["-O2", "-g0"], "nvcc": ["-O2"]},
    ))

setup(
    name="pcdet-clean",
    version="0.5.2",
    packages=find_namespace_packages(include=["pcdet", "pcdet.*"]),
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
