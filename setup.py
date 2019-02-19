import os
import re
import sys
import platform
import subprocess
import unittest

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

PACKAGE_NAME = "qflow"
BACKEND_NAME = f"_{PACKAGE_NAME}_backend"


def load_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")
    return test_suite


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        env = os.environ.copy()

        cfg = env.get("CMAKE_BUILD_TYPE", "Debug" if self.debug else "Release")
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j5", BACKEND_NAME]
            if cfg.lower() == "coverage":
                build_args += ["gtest"]

        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        print(f"Running build with this env: {env}")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            " ".join(["cmake", ext.sourcedir] + cmake_args),
            cwd=self.build_temp,
            env=env,
            shell=True,
        )
        subprocess.check_call(
            " ".join(["cmake", "--build", "."] + build_args),
            cwd=self.build_temp,
            env=env,
            shell=True,
        )


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version="0.1.0",
        author="Bendik Samseth",
        author_email="b.samseth@gmail.com",
        description="Variational Monte Carlo Framework",
        long_description="",
        ext_modules=[CMakeExtension(BACKEND_NAME)],
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False,
        test_suite="setup.load_test_suite",
        install_requires=[
            "mpi4py",
            "numpy",
            "autograd",
            "matplotlib",
            "scipy",
            "seaborn",
        ],
    )
