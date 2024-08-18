# Solid Conversation

A Python library for generating intent-aware dialogues using large language models. More details are available on (https://arxiv.org/abs/2402.11633)[paper] and its (https://github.com/arian-askari/solid)[corresponding repo].

## Installation

- Clone the code.
- ```cd SOLID_CONV```
- ``` pip install -e . ```



## Usage

```python
from solid_conversation import ConversationGenerator

generator = ConversationGenerator("HuggingFaceH4/zephyr-7b-beta")
dialogue, instructions = generator.generate_dialogue("topic", "type", "context", ["OQ", "CQ"], "first question")
print(dialogue)



