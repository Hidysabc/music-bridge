from setuptools import setup
from setuptools import find_packages

setup(name='musicbridge',
      version='0.0.1',
      description='Music mixer using deep neural networks',
      author=['Wei-Yi Cheng', 'Hidy Chiu'],
      author_email=['ninpy.weiyi@gmail.com', 'hidy0503@gmail.com'],
      #url='https://github.com/fchollet/keras',
      #download_url='https://github.com/fchollet/keras/tarball/2.0.2',
      license='GPLv3',
      install_requires=['Keras==2.0.2', 'boto3', 'pandas'],
      extras_require={
          'h5py': ['h5py'],
      },
      packages=find_packages())
