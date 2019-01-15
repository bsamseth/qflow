import cppyy
import tempfile
import os
import glob
import numpy as np

install_dir = tempfile.mkdtemp(prefix='qmc')
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
source_dir = os.path.dirname(current_dir)
install_dir = os.path.join(source_dir, 'build')
include_dir = os.path.join(source_dir, 'include')
eigen_dir = os.path.join(source_dir, 'external', 'eigen')
print(current_dir, source_dir, include_dir, install_dir)

def cmake_run(build_type='Release', c_compiler='gcc', cxx_compiler='g++'):
    os.environ['CC'] = c_compiler
    os.environ['CXX'] = cxx_compiler
    os.system('cd {} && cmake {} -DCMAKE_BUILD_TYPE={}'.format(install_dir, source_dir, build_type))

def load_library():
    os.system('cd {} && make engine'.format(install_dir))
    libraries = glob.glob(os.path.join(install_dir, 'libengine.*'))
    print('Found libraries: {}'.format(libraries))
    library = libraries[-1]
    cppyy.add_include_path(eigen_dir)
    cppyy.load_library(library)
    for header in glob.glob(os.path.join(include_dir, '*.hpp')):
        print('Loading {}'.format(header))
        cppyy.include(header)



cmake_run()
load_library()

from cppyy.gbl import rand_gen, Vector, System, RBMHarmonicOscillatorHamiltonian, RBMInteractingHamiltonian, RBMWavefunction, MetropolisSampler, ImportanceSampler, GibbsSampler, SgdOptimizer, AdamOptimizer


def array_to_vector(arr):
    v = Vector(len(arr))
    for i, a in enumerate(arr):
        v[i] = a
    return v
def vector_to_array(vec):
    return np.asarray([vec[i] for i in range(vec.size())])

def array_to_system(arr):
    s = System(arr.shape[0], arr.shape[1])
    for i, a in enumerate(arr):
        for j, b in enumerate(a):
            s(i)[j] = b
    return s

def system_to_array(system):
    return np.asarray([vector_to_array(system[i]) for i in range(system.cols())])