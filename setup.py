from setuptools import setup, find_packages

setup(
    name='ordbetareg',
    version='0.1',
    description='Ordered Beta Regression Model with scipy',
    author='Robert Kubinec',
    author_email='rkubinec@mailbox.sc.edu',
    url='https://github.com/saudiwin/ordbetareg_py',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
