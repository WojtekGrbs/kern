import os
import sys
import tempfile

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def libomp_prefix():
    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.exists(os.path.join(prefix, "include", "omp.h")):
            return prefix
    return None


def openmp_flags():
    if sys.platform == "win32":
        return ["/openmp"], []

    if sys.platform == "darwin":
        prefix = libomp_prefix()
        if not prefix:
            return None

        return (
            [
                "-Xpreprocessor",
                "-fopenmp",
                f"-I{os.path.join(prefix, 'include')}",
            ],
            [
                f"-L{os.path.join(prefix, 'lib')}",
                "-lomp",
            ],
        )

    return ["-fopenmp"], ["-fopenmp"]


class BuildExt(build_ext):
    def has_openmp(self, compile_args, link_args):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "test_openmp.c")

            with open(source, "w", encoding="utf-8") as f:
                f.write(
                    "#include <omp.h>\n"
                    "int main(void) { return omp_get_max_threads() < 1; }\n"
                )

            try:
                objects = self.compiler.compile(
                    [source],
                    output_dir=tmp,
                    extra_postargs=compile_args,
                )
                self.compiler.link_executable(
                    objects,
                    os.path.join(tmp, "test_openmp"),
                    extra_postargs=link_args,
                )
                return True
            except Exception:
                return False

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/O2"]
            link_args = []
        else:
            compile_args = ["-O3", "-ffast-math", "-fno-math-errno"]

            if sys.platform == "darwin":
                compile_args += ["-mcpu=native", "-fveclib=Accelerate"]
            else:
                compile_args.append("-march=native")

            link_args = []

        omp = openmp_flags()

        if omp is not None:
            omp_compile_args, omp_link_args = omp

            if self.has_openmp(omp_compile_args, omp_link_args):
                compile_args += omp_compile_args
                link_args += omp_link_args
            else:
                print("OpenMP was not found; building the serial fallback.")
        else:
            print("OpenMP was not found; building the serial fallback.")

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args

        super().build_extensions()


ext = Extension(
    "kern._core",
    sources=[
        "kern/kde.c",
        "kern/kernels.c",
        "kern/module.c",
        "kern/bandwidth.c",
    ],
    include_dirs=["kern", numpy.get_include()],
)


setup(
    name="kern",
    version="0.1",
    packages=["kern"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
)