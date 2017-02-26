from setuptools import find_packages, setup

setup(
    name='block',
    version='0.0.2',
    description="Improved block matrix creation for numpy and PyTorch.",
    author='Brandon Amos',
    author_email='bamos@cs.cmu.edu',
    platforms=['any'],
    license="Apache 2.0",
    url='https://github.com/bamos/block',
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
    ]
)
