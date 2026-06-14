import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).parent.resolve()


DESCRIPTION = "kernel density estimation backed by a C extension" # if README.md is missing or empty
CONFIG_FIELDS = "define_macros include_dirs library_dirs libraries extra_compile_args extra_link_args".split()
CBLAS_MACRO = [("USE_CBLAS", "1")]

# C programs used to test whether OpenMP / CBLAS actually compile and link
OPENMP_PROBE = "#include <omp.h>\nint main(void) { return omp_get_max_threads() < 1; }\n"
BLAS_PROBE = """#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
int main(void) {
    double x[1] = {1.0};
    return cblas_ddot(1, x, 1, x, 1) != 1.0;
}
"""


def read_long_description():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").strip()
    return readme or DESCRIPTION


def option(name):
    # Reads an environment variable like KERN_USE_OPENMP and normalizes it
    # into one of three states: "auto", "required", or "disabled"
    value = os.environ.get(name, "auto").strip().lower()
    for result, aliases in {
        "auto": {"auto", ""},
        "required": {"1", "true", "yes", "on", "required"},
        "disabled": {"0", "false", "no", "off", "disabled"},
    }.items():
        if value in aliases:
            return result
    raise RuntimeError(
        f"{name} must be auto, required/1, or disabled/0; got {value!r}"
    )


def env_args(name, *, path=False, shell=False, commas=False):
    #   path=True   -> split on the OS path separator (":" or ";")
    #   shell=True  -> split like a shell command line (respects quotes)
    #   commas=True  -> treat commas as separators too
    value = os.environ.get(name, "")
    if shell:
        return shlex.split(value, posix=os.name != "nt") if value else []
    if commas:
        value = value.replace(",", " ")
    return [item for item in value.split(os.pathsep if path else None) if item]


def pkg_config(package):
    # Tries to ask the system's pkg-config tool for compile/link flags
    # for a given library. Returns None on error.
    def run(flag):
        return subprocess.run(
            ["pkg-config", flag, package],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    try:
        return {
            "extra_compile_args": shlex.split(run("--cflags")),
            "extra_link_args": shlex.split(run("--libs")),
        }
    except:
        return None


def config(name, **values):
    return {"name": name, **{field: [] for field in CONFIG_FIELDS}, **values}


def has_header(prefix, *parts):
    # Checks whether prefix/include/<parts...> exists
    # used to check if a library's header file (like omp.h or cblas.h) is present
    return prefix and os.path.exists(os.path.join(prefix, "include", *parts))


def libomp_prefix():
    # Tries to find where libomp (OpenMP for macOS/clang) is installed,
    # by checking an env var first, then Homebrew
    return next(
        (
            prefix
            for prefix in (
                os.environ.get("KERN_LIBOMP_PREFIX"),
                "/opt/homebrew/opt/libomp",
                "/usr/local/opt/libomp",
            )
            if has_header(prefix, "omp.h")
        ),
        None,
    )


def openmp_config():
    # Returns a candidate config for enabling OpenMP, depending on platform.
    # Returns None if we don't know how to enable OpenMP here.

    ## WINDOWS 
    if sys.platform == "win32":
        # MSVC uses a simple /openmp flag
        return config("MSVC OpenMP", extra_compile_args=["/openmp"])

    ## NON-MACOS
    if sys.platform != "darwin":
        return config(
            "compiler OpenMP",
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
        )

    ## MACOS
    # Apple's default clang doesn't ship OpenMP, so we need libomp
    prefix = libomp_prefix()
    if not prefix:
        return None

    library_dir = os.path.join(prefix, "lib")
    return config(
        "libomp",
        extra_compile_args=[
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{os.path.join(prefix, 'include')}",
        ],
        extra_link_args=[f"-L{library_dir}", "-lomp", f"-Wl,-rpath,{library_dir}"],
    )


def blas_candidates():
    # Yields a series of possible ways to enable CBLAS

    # 1. Fully custom config via environment variables (highest priority)
    custom_values = {
        "include_dirs": env_args("KERN_BLAS_INCLUDE_DIRS", path=True),
        "library_dirs": env_args("KERN_BLAS_LIBRARY_DIRS", path=True),
        "libraries": env_args("KERN_BLAS_LIBRARIES", commas=True),
        "extra_compile_args": env_args("KERN_BLAS_COMPILE_ARGS", shell=True),
        "extra_link_args": env_args("KERN_BLAS_LINK_ARGS", shell=True),
    }
    if any(custom_values.values()):
        yield config("custom CBLAS", define_macros=CBLAS_MACRO, **custom_values)

    if sys.platform == "darwin":
        # 2. macOS: try Apple's built-in Accelerate framework first
        yield config(
            "Accelerate",
            define_macros=CBLAS_MACRO + [("USE_ACCELERATE", "1")],
            extra_link_args=["-framework", "Accelerate"],
        )
        # 3. macOS: try Homebrew-installed OpenBLAS
        for prefix in ("/opt/homebrew/opt/openblas", "/usr/local/opt/openblas"):
            if has_header(prefix, "cblas.h"):
                yield config(
                    f"OpenBLAS at {prefix}",
                    define_macros=CBLAS_MACRO,
                    include_dirs=[os.path.join(prefix, "include")],
                    library_dirs=[os.path.join(prefix, "lib")],
                    libraries=["openblas"],
                )

    # 4. Try pkg-config for common BLAS implementations
    for package in ("openblas", "cblas", "blas"):
        flags = pkg_config(package)
        if flags:
            yield config(f"pkg-config {package}", define_macros=CBLAS_MACRO, **flags)

    # 5. try linking against the library by name directly
    for library in ("openblas", "cblas", "blas"):
        yield config(library, define_macros=CBLAS_MACRO, libraries=[library])


class BuildExt(build_ext):
    def probe(self, source, candidate):
        # Tries to compile and link the C programs (source) using
        # tcurrent "candidate".
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp, "probe.c")
            source_path.write_text(source, encoding="utf-8")

            try:
                objects = self.compiler.compile(
                    [str(source_path)],
                    output_dir=tmp,
                    macros=candidate["define_macros"],
                    include_dirs=candidate["include_dirs"],
                    extra_postargs=candidate["extra_compile_args"],
                )
                self.compiler.link_executable(
                    objects,
                    os.path.join(tmp, "probe"),
                    libraries=candidate["libraries"],
                    library_dirs=candidate["library_dirs"],
                    extra_postargs=candidate["extra_link_args"],
                )
                return True # If we are here it means that the 
            except Exception:
                return False

    @staticmethod
    def apply_config(extension, candidate):
        # Copies all the settings from "candidate"
        # onto the real Extension object, appending to whatever
        # was already there
        for field in CONFIG_FIELDS:
            setattr(extension, field, list(getattr(extension, field) or []) + candidate[field])

    def configure_optional(self, extension, *, mode, candidates, source, enabled, required, fallback):
        # Generic helper used for both OpenMP and BLAS:


        # - if not disabled, try each candidate config by probing it
        # - on the first one that works, apply it and report success


        # - if none work and mode is "required", raise an error
        # - otherwise just print a "feature unavailable, using fallback" message
        if mode != "disabled":
            for candidate in candidates:
                if candidate and self.probe(source, candidate):
                    self.apply_config(extension, candidate)
                    self.announce(enabled.format(name=candidate["name"]), level=2)
                    return
        if mode == "required":
            raise RuntimeError(required)
        self.announce(fallback, level=2)

    def configure_openmp(self, extension):
        # Decide whether/how to enable OpenMP for this extension
        self.configure_optional(
            extension,
            mode=option("KERN_USE_OPENMP"),
            candidates=(openmp_config(),),
            source=OPENMP_PROBE,
            enabled="OpenMP enabled with {name}",
            required=(
                "KERN_USE_OPENMP is required, but OpenMP could not be compiled "
                "and linked. Install an OpenMP runtime or disable this option."
            ),
            fallback="OpenMP unavailable; building the serial fallback",
        )

    def configure_blas(self, extension):
        # Decide whether/how to enable CBLAS for this extension
        self.configure_optional(
            extension,
            mode=option("KERN_USE_BLAS"),
            candidates=blas_candidates(),
            source=BLAS_PROBE,
            enabled="CBLAS enabled with {name}",
            required=(
                "KERN_USE_BLAS is required, but no usable CBLAS implementation "
                "was found. Install OpenBLAS, use macOS Accelerate, or provide "
                "KERN_BLAS_* build flags."
            ),
            fallback="CBLAS unavailable; building the scalar multivariate fallback",
        )

    def build_extensions(self):
        # Called by setuptools right before compiling the extension(s).
        # We set base optimization flags, then probe for OpenMP and BLAS
        # support and adjust the extension's build settings accordingly.
        for extension in self.extensions:
            if self.compiler.compiler_type == "msvc":
                # Windows/MSVC: enable optimizations and define "restrict"
                # since MSVC doesn't support the C "restrict" keyword natively
                extension.extra_compile_args = ["/O2"]
                extension.define_macros = [
                    *list(extension.define_macros or []),
                    ("restrict", "__restrict"),
                ]
            else:
                # Other compilers (gcc/clang): standard optimization flag
                extension.extra_compile_args = ["-O3", "-fno-math-errno", "-march=native",]

            extension.extra_link_args = []
            self.configure_openmp(extension)
            self.configure_blas(extension)

        # Let setuptools do the actual compiling/linking
        super().build_extensions()

ext = Extension(
    "kern._core",
    sources=[
        "kern/kde.c",
        "kern/kernels.c",
        "kern/module.c",
        "kern/bandwidth.c",
        "kern/kde_approx.c",
        "kern/kde_multivariate.c",
    ],
    depends=[
        "kern/bandwidth.h",
        "kern/kde.h",
        "kern/kde_approx.h",
        "kern/kde_multivariate.h",
        "kern/kernels.h",
    ],
    include_dirs=["kern", numpy.get_include()],
)

setup(
    name="kern",
    version="0.2.0",
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Wojciech Grabias",
    url="https://github.com/WojtekGrbs/kern",
    project_urls={
        "Documentation": "https://wojtekgrbs.github.io/kern/",
        "Source": "https://github.com/WojtekGrbs/kern",
    },
    license="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    extras_require={"docs": ["sphinx>=7", "furo>=2024.1.29"], "test": ["pytest>=8"]},
    packages=find_packages(),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering",
    ],
)