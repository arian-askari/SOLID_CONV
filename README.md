# A Python library for Intent-Aware Conversation Generation with SOLID 
[<img src= "./figures/solid_logov5.png" width=95px />](https://arian-askari.github.io/SOLID/) [<img src= "https://img.shields.io/badge/Language-English-brightgreen"  />](https://arian-askari.github.io/SOLID/)

A Python library for generating intent-aware dialogues using large language models. More details are available on "[SOLID: Self-instructing and Self-seeding LLMs for Large-scale Intent-Aware Informating-Seeeking Dialogue Generation](https://arxiv.org/abs/2402.11633)" and its (corresponding repo)[https://github.com/arian-askari/solid].

# Self-seeding and Multi-intent Self-instructing LLMs for Generating Intent-aware Information-Seeking dialogs

If you use this dataset, please use the following bibtex references:


```bibtex
@misc{askari2024selfseeding,
      title={Self-seeding and Multi-intent Self-instructing LLMs for Generating Intent-aware Information-Seeking dialogs}, 
      author={Arian Askari and Roxana Petcu and Chuan Meng and Mohammad Aliannejadi and Amin Abolghasemi and Evangelos Kanoulas and Suzan Verberne},
      year={2024},
      eprint={2402.11633},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This work has been done under the supervision of Prof. Mohammad Aliannejadi, Evangelos Kanoulas, and Suzan Verberne during my visiting research at Information Retrieval Lab at the University of Amsterdam (IRLab@UvA).

## Overiew
We introduce SOLID, a novel approach to generating large-scale, intent-aware information-seeking dialogues. Our method leverages self-seeding and multi-intent self-instructing schemes to produce high-quality dialogues. Additionally, we propose SOLID-RL, an enhanced version of SOLID, designed to increase efficiency without compromising the quality of generated dialogues. SOLID's method to creating intent-aware dialogs highlights the possibilities of using LLMs to generate meaningful and diverse while intent-aware conversations.


## Installation

- Clone the code.
- ```cd SOLID_CONV```
- ``` pip install -e . ```



## Usage

```python
import pprint
from solid_conversation import ConversationGenerator

entity_name = "Jönköping"
entity_type = "City"
entity_description = ("Jönköping (Swedish pronunciation: [jœnˈøːnpɪŋ] (listen)) is a city in southern Sweden, situated by the western shore of Lake Vättern. "
                      "It is the seat of Jönköping municipality and Jämtland County, and has a population of 114,418 (2019). Jönköping is part of the Swedish province "
                      "of Småland. Historically, the city has been significant due to its location at the transition between the provinces of Västergötland and Småland, "
                      "which is reflected in its architecture and cultural heritage.")
sequence_of_intents = ['OQ', 'RQ', 'FD_NF', 'PA']
conversation_starter = "How has the cultural and architectural heritage of both Västergötland and Småland influenced the development of Jönköping as a unique city?"
solid_generated_dialogue, multi_intent_self_instructions = generator.generate_dialogue(entity_name, entity_type, entity_description, sequence_of_intents, conversation_starter)

dialog = solid_generated_dialogue["generated_dialogue"]
print(dialog)
pprint.pprint(dialog, width=80)

```

## Colab notebook

[Intent-aware-Conv-Generation-SOLID](https://colab.research.google.com/drive/1Roohw7CVrsSLyvYNedPEowjHOwSYPRTt?usp=sharing) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Roohw7CVrsSLyvYNedPEowjHOwSYPRTt?usp=sharing)
