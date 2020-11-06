from setuptools import setup

setup(
    name='pyHIIExplorer',
    version='0.1.0',    
    description='A new version of HIIExplorer for python',
    url='https://github.com/sfsanchez72/pyHIIExplorer',
    author='Sebastian F. Sanchez & Alejandra Z. Lugo',
    author_email='sfsanchez@astro.unam.mx',
    license='BSD 2-clause',
    packages=['pyHIIExplorer'],
    install_requires=['numpy',
                      'scipy',
                      'astropy',
                      'matplotlib',
                      'scikit-image',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
