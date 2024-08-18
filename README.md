# SOLID: Intent-Aware Conversation Generation 
[<img src= "./figures/solid_logo.png" width=95px />](https://arian-askari.github.io/SOLID/) [<img src= "https://img.shields.io/badge/Language-English-brightgreen"  />](https://arian-askari.github.io/SOLID/)


Welcome to SOLID-Conv – a Python library designed for generating intent-aware dialogues using large language models. Discover more about SOLID in our [research paper](https://arxiv.org/abs/2402.11633), which has been accepted at the **ACL 2024** NLP for Conversational AI workshop, and explore the [GitHub repository](https://github.com/arian-askari/solid) for the latest updates.

## Overview

SOLID introduces a novel methodology for generating large-scale, intent-aware information-seeking dialogues. By leveraging self-seeding and multi-intent self-instructing techniques, SOLID ensures high-quality, diverse, and meaningful conversations. SOLID-RL, an advanced variant of SOLID, enhances efficiency while maintaining dialogue quality.

Our approach demonstrates the power of using large language models (LLMs) to create dialogues that are not only informative but also contextually rich and diverse.

## Explore in Google Colab

Try out SOLID in a ready-to-use Google Colab notebook and see it in action! 


[Intent-aware-Conv-Generation-SOLID](https://colab.research.google.com/drive/1Roohw7CVrsSLyvYNedPEowjHOwSYPRTt?usp=sharing) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Roohw7CVrsSLyvYNedPEowjHOwSYPRTt?usp=sharing)

## Figure of SOLID's pipeline
<details>
<summary>Click to expand the SOLID pipeline figure</summary>



<img src="./figures/SOLID_pipeline.svg" alt="SOLID Pipeline" width="800">

</details>


## Installation
<details>
<summary>Click to expand Installation</summary>

### To get started with SOLID, follow these steps:

### Download the latest version:

``````bash
wget https://github.com/arian-askari/SOLID_CONV/archive/refs/heads/main.zip
``````

### Unzip the downloaded file:
``````bash
unzip main.zip
``````

### Move the files to the current directory:

``````bash
mv ./SOLID_CONV-main/* ./
``````

### Install the package:

``````bash
pip install ./
``````

</details>



## Usage

### Here's a quick example to get you started with SOLID:

``````python
import pprint
from solid_conversation import ConversationGenerator

# Initialize the conversation generator with your chosen LLM model
llm_model_name = "HuggingFaceH4/zephyr-7b-beta"
generator = ConversationGenerator(llm_model_name)

# A self-seed example
entity_name = "Jönköping"
entity_type = "City"
entity_description = ("Jönköping (Swedish pronunciation: [jœnˈøːnpɪŋ] (listen)) is a city in southern Sweden, situated by the western shore of Lake Vättern. "
                      "It is the seat of Jönköping municipality and Jämtland County, and has a population of 114,418 (2019). Jönköping is part of the Swedish province "
                      "of Småland. Historically, the city has been significant due to its location at the transition between the provinces of Västergötland and Småland, "
                      "which is reflected in its architecture and cultural heritage.")
sequence_of_intents = ['OQ', 'RQ', 'FD_NF', 'PA']
conversation_starter = "How has the cultural and architectural heritage of both Västergötland and Småland influenced the development of Jönköping as a unique city?"

# Generate the dialogue
solid_generated_dialogue, multi_intent_self_instructions = generator.generate_dialogue(entity_name, entity_type, entity_description, sequence_of_intents, conversation_starter)

# Print the generated dialogue
dialog = solid_generated_dialogue["generated_dialogue"]
pprint.pprint(dialog, width=80)
``````



## Citing SOLID

If you use SOLID in your research, please cite our work:

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

## Available Intents
<details>
<summary>Click to expand the Available Intents</summary>

### SOLID supports a variety of intents to generate different types of dialogue responses. Below is a list of the available intents, including their purpose and how they should be used:

- **CQ**: Clarification Question
- **FD**: Further Details
- **GG**: Gratitude
- **PA**: Potential Answer
- **IR**: Information Request
- **OQ**: Original Question
- **FQ**: Follow-up Question
- **RQ**: Rephrased Question
- **PF**: Positive Feedback
- **NF**: Negative Feedback
- **JK**: Jabberwocky
- **O**: System Error

You can combine these intents for dialogue generation. For example, using NF_FQ would create an utterance in the dialog that includes both negative feedback and a follow-up question. Explore the various intents to create dynamic and engaging interactions with SOLID!

### Custom intents

#### Example of defining a new custom intents_dict

```python
custom_intents_dict = {
    "OQ": {
        "user instruction": "Formulate the first question posed by a user that initiates a QA dialogue.",
        "agent instruction": "Formulate an original question posed by an agent.",
        "user generation": "Question:",
        "agent generation": "Question:",
    },
    # Add or update more intents as needed
}

generator.set_intents_dict(custom_intents_dict)

solid_generated_dialogue, multi_intent_self_instructions = generator.generate_dialogue(entity_name, entity_type, entity_description, sequence_of_intents, conversation_starter)
dialog = solid_generated_dialogue["generated_dialogue"]
```

#### Example of adding a new intent to the existing intents_dict

```python
generator.add_intent(
    "NEW_INTENT",
    "New user instruction",
    "New agent instruction",
    "New user generation",
    "New agent generation"
)

solid_generated_dialogue, multi_intent_self_instructions = generator.generate_dialogue(entity_name, entity_type, entity_description, sequence_of_intents, conversation_starter)
dialog = solid_generated_dialogue["generated_dialogue"]
```
</details>

## Acknowledgments

This project was developed under the guidance of Prof. Mohammad Aliannejadi, Evangelos Kanoulas, and Suzan Verberne during my research visit to the Information Retrieval Lab at the University of Amsterdam (IRLab@UvA).


