from setuptools import setup

readme = open('README.rst').read()

setup(name='aiutils',
      version='0.1.0',
      description='Supporting utilities for dtoolAI',
      long_description=readme,
      long_description_content_type='text/x-rst',
      url='http://github.com/JIC-CSB/aiutils',
      author='Matthew Hartley',
      author_email='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      install_requires=[
        'dtoolai',
      ],
      packages=['aiutils'],
      zip_safe=False)
