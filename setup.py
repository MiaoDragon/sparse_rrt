# adapted from https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936

import os
import re
import sys
import sysconfig
import platform
import subprocess
from pathlib import Path

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from subprocess import Popen, PIPE, STDOUT

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'], stderr=STDOUT)
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ['--', '-j2', '-std=c++1y']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-'*10, 'Running CMake prepare', '-'*40)
        #subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
        #                      cwd=self.build_temp, env=env)
        # printout the output
        p = subprocess.Popen(['cmake', cmake_list_dir] + cmake_args, cwd=self.build_temp, env=env, stdout=PIPE,
                              bufsize=1, universal_newlines=True)
        for line in p.stdout:
            print(line) # process line here
        p.wait()

        print('-'*10, 'Building extensions', '-'*40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        #subprocess.check_call(cmake_cmd,
        #                      cwd=self.build_temp)
        p = subprocess.Popen(cmake_cmd, cwd=self.build_temp, stdout=PIPE,
                              bufsize=1, universal_newlines=True)
        for line in p.stdout:
            print(line) # process line here
        p.wait()


        print('finished cmake build.')
        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


ext_modules = [
  CMakeExtension('sparse_rrt._sst_module')
]

setup(
 # ...
  packages=find_packages(),
  ext_modules=ext_modules,
  cmdclass=dict(build_ext=CMakeBuild),
  zip_safe=False,
)
