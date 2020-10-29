import setuptools


with open('requirements.txt') as f:
    required = f.read().splitlines()


setuptools.setup(
    name='dtl_nlp',
    version='2020.10.29',
    author='timniven',
    author_email='tim.niven.public@gmail.com',
    description='Deep learning tools.',
    url='https://github.com/timniven/hsdl',
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=required
)
