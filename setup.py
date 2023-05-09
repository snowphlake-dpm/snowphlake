import setuptools

setuptools.setup(
    name="snowphlake",
    version="1.0.0",
    author="Vikram Venkatraghavan",
    author_email="v.venkatraghavan@amsterdamumc.nl",
    description="Personalized Disease Progression Model",
    long_description="Staging NeurOdegeneration With PHenotype informed progression timeLine of biomarKErs",
    long_description_content_type="text/markdown",
    url="https://github.com/88vikram/snowphlake",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
      'numpy',
      'pandas',
      'scikit-learn',
      'scipy',
      'matplotlib',
      'rpy2',
      'nimfa',
      'pathos',
      'plotly'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
