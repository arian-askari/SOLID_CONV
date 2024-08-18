from setuptools import setup, find_packages

setup(
    name="solid_conversation",
    version="0.1.0",
    description="A library for generating intent-aware dialogues using LLMs.",
    author="Arian Askari, Roxana Petcu",
    packages=find_packages(),
    install_requires=[
        "bitsandbytes==0.43.3",
        "transformers @ git+https://github.com/huggingface/transformers.git@52cb4034ada381fe1ffe8d428a1076e5411a8026",
        "peft @ git+https://github.com/huggingface/peft.git@e8ba7de5732e050a8f061cde555c02ab575f7529",
        "accelerate @ git+https://github.com/huggingface/accelerate.git@589fddd317f008e704073c133bc2cb8958f287e6",
        "tqdm",
    ],
    python_requires=">=3.10, <3.13",
)
