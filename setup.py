import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='chemrecommender',
    version='0.0.3',
    author="DRP Project",
    author_email="darkreactionproject@haverford.edu",
    description="A standalone module to build a recommender pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darkreactions/recommendation_engine",
    packages=['chemrecommender'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas', 'chemdescriptor'],
    include_package_data=True
)
