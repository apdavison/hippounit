try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='hippounit',
    version='0.2.0dev',
    author='Sara Saray, Christian Roessert, Andrew Davison',
    author_email='andrew.davison@unic.cnrs-gif.fr',
    packages=['hippounit', 'hippounit.tests'],
    url='http://github.com/apdavison/hippounit',
    license='MIT',
    description='A SciUnit library for data-driven validation testing of models of hippocampus.',
    long_description="",
    install_requires=['sciunit>=0.1.3.1', 'neuronunit']
)
