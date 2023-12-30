from setuptools import setup, find_packages

setup(
    name="neura_command",  # Use underscores instead of hyphens
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.25.2",
    ],
    # Additional metadata
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # If your README is Markdown
    url="https://github.com/yourusername/neura_command",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Add other metadata as needed
)
