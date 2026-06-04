import os
import sys
import tempfile

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def libomp_prefix():
    env_prefix = os.environ.get("KERN_LIBOMP_PREFIX")
    if env_prefix:
        return env_prefix

    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.exists(os.path.join(prefix, "include", "omp.h")):
            return prefix

    return None


def openmp_flags(platform):
    if platform == "win32":
        return ["/openmp"], []
    if platform == "darwin":
        prefix = libomp_prefix()
        compile_args = ["-Xpreprocessor", "-fopenmp"]
        link_args = ["-lomp"]
        if prefix:
            compile_args.append(f"-I{os.path.join(prefix, 'include')}")
            link_args.insert(0, f"-L{os.path.join(prefix, 'lib')}")
        return compile_args, link_args
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
        use_openmp = os.environ.get("KERN_DISABLE_OPENMP") != "1"
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/O2"]
        else:
            compile_args = ["-O3", "-ffast-math", "-fno-math-errno"]
            if sys.platform == "darwin":
                compile_args += ["-mcpu=native", "-fveclib=Accelerate"]
            else:
                compile_args.append("-march=native")
        link_args = []

        if use_openmp:
            force_openmp = os.environ.get("KERN_USE_OPENMP") == "1"
            can_try_openmp = sys.platform != "darwin" or force_openmp or libomp_prefix()

            if can_try_openmp:
                omp_compile, omp_link = openmp_flags(sys.platform)
                if self.has_openmp(omp_compile, omp_link):
                    compile_args += omp_compile
                    link_args += omp_link
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
    sources=["kern/kde.c",
             "kern/kernel.c"],
    include_dirs=["kern", numpy.get_include()],
)

setup(
    name="kern",
    version="0.1",
    packages=["kern"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
)
