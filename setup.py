from setuptools import setup, find_packages

long_description = """A package for data manipulation, data preprocessing, data
visualization in computer vision projects, particularly Tensorflow's Object
Detection API-based projects. Supports multiple advanced techniques for data
preprocessing and especially can be used seamlessly with Object Detection API."""

setup(
    name="vebits_api",
    version="1.1.5",
    author="Hoang Nghia Tuyen",
    author_email="hnt4499@gmail.com",
    url="https://github.com/hnt4499/vebits_api",
    download_url="https://github.com/hnt4499/vebits_api/archive/1.0.tar.gz",
    install_requires=["pandas", "numpy", "tqdm", "imutils", "protobuf",
                      "matplotlib", "imgaug", "opencv-python"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md", "requirements.txt"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="High-level deep learning package for Object Detection API",
    long_description=long_description,
    keywords=["augmentation", "image", "deep learning", "neural network", "CNN",
              "machine learning", "computer vision", "object detection api",
              "tensorflow", "data preprocessing", "data manipulation",
              "data visualization"]
)
