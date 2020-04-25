from distutils.core import setup

setup(
    name='BioProc',
    version='0.1dev',
    description='Biological signal processing for EMG, ECG and EEG data.',
    autor='Praveen Prabhakar KR',
    author_email='praveenp@msu.edu',
    packages=['bioproc',],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'matplotlib',
        'autopep8',
        'scipy',
        'pylint',
        'pytest',
        'pdoc3',
        'biosppy',
        'tensorflow',
        'keras',
        'wfdb',
    ])