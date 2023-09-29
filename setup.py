# Количество разработчиков - 1
# Количество веток - 1
# Количество копирований - 3
# Регулярность использования - 2

from setuptools import setup
from io import open

setup(
    name='financial_readability',
    packages=['financial_readability'],
    version='0.1.0',
    description='Calculate readability measure from text-paragraph',
    author='Linus Graf',
    author_email='linusgraf92@gmail.com',
    url='https://github.com/grooof/financial_readability',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    package_data={'word_lists': ['word_lists'],'models': ['models']},
    include_package_data=True,
    install_requires=['textstat','pyphen', 'nltk', 'sklearn', 
    'pkg_resources', 'ast', 'spacy', 'numpy', 'pandas', 'pickle'],
    license='MIT'
)
