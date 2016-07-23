from distutils.core import setup

try:
    from sphinx.setup_command import BuildDoc
    cmdclass={'build_sphinx':BuildDoc}
except ImportError:
    cmdclass={}

setup(name='ismrmrd-python-tools',
        version='0.3',
        description='ISMRMRD Image Reconstruction Tools',
        author='ISMRMRD Developers',
        url='https://github.com/ismrmrd/ismrmrd-python-tools',
        packages=['ismrmrdtools'],
        cmdclass=cmdclass
        )
