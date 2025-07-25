import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setuptools.setup(
    name="snowphlake",
    version="1.0.0",
    author="Vikram Venkatraghavan",
    author_email="v.venkatraghavan@amsterdamumc.nl",
    description="Personalized Disease Progression Model",
    long_description="Staging NeurOdegeneration With PHenotype informed progression timeLine of biomarKErs",
    long_description_content_type="text/markdown",
    url="https://github.com/snowphlake-dpm/snowphlake",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
