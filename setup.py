from setuptools import find_packages, setup

__version__ = '0.1'
url = 'https://github.com/ntt123/haiku_trainer'
download_url = '{}/archive/{}.tar.gz'.format(url, __version__)

install_requires = []
setup_requires = []
tests_require = []

setup(
    name='haiku_trainer',
    version=__version__,
    description='A helper library for training dm-haiku models.',
    author='Thông Nguyên',
    author_email='xcodevn@gmail.com',
    url=url,
    download_url=download_url,
    keywords=['dm-haiku', 'parameters', 'deep-learning', 'trainer', 'jax'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
