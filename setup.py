from setuptools import setup, find_packages
from pathlib import Path
# Read README.md as long_description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='aireadi_loader',
    version='0.1.0',
    author='Yuka Kihara et al.',
    author_email='yk73@uw.edu',
    description='Dataloader for the AIREADI dataset.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/uw-biomedical-ml/aireadi_loader',
    license='BSD-2-Clause',
    packages=find_packages(include=['aireadi_loader', 'aireadi_loader.*']),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'tqdm',
        'scikit-learn',
        'opencv-python',
        'matplotlib',
        'Pillow',
        'monai',
        'pydicom',
        # add more if needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD 2-Clause',
    ],
    python_requires='>=3.7',
)
