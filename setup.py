import setuptools
with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='waddle',
    version='0.1',
    scripts=['waddle/embedding.py'],
    author='Andrew Maher',
    author_email='andrewtmmaher@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andrewtmmaher/waddle',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)