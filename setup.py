from setuptools import setup, find_packages

setup(
    name="solid_conversation",
    version="0.1.0",
    description="A library for generating intent-aware dialogues using LLMs.",
    author="Arian Askari, Roxana Petcu",
    packages=find_packages(),
    install_requires=[
        "transformers",  # You can add other dependencies as needed
        "tqdm",
    ],
    python_requires=">=3.7",
)
