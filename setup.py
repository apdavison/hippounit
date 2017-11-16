try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

json_files = package_files('hippounit/tests/somafeat_stim')

setup(
    name='hippounit',
    version='0.3.0dev',
    author='Sara Saray, Christian Rössert, Andrew Davison, Shailesh Appukuttan',
    author_email='andrew.davison@unic.cnrs-gif.fr, shailesh.appukuttan@unic.cnrs-gif.fr',
    packages=['hippounit', 'hippounit.tests', 'hippounit.capabilities', 'hippounit.scores'],
    package_data={'hippounit': json_files},
    url='http://github.com/apdavison/hippounit',
    license='MIT',
    description='A SciUnit library for data-driven validation testing of models of hippocampus.',
    long_description="",
    install_requires=['sciunit>=0.1.3.1', 'neuronunit']
)
