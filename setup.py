from setuptools import setup, find_packages

setup(
    name='CosTaL',
    version='0.0.2.1',
    url='https://github.com/li000678/CosTaL',
    license='MIT',
    author='Yijia Li',
    author_email='li000678@umn.edu',
    description='An accurate and scalable graph-based clustering algorithm for high-dimensional single-cell data analysis',
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    package_data={
        '':['p_knng_Linux'],
               },
    install_requires=[
        'numpy',
        'pandas',
        'igraph',
        'leidenalg',
        'numba',
        'scipy',
        'sklearn',
        'phenograph', # Alpha stage only
    ],
)
