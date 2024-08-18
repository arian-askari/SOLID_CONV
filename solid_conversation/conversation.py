import torch
from .utils import (
    filter_new_turn,
    trim_to_last_punctuation,
    get_turn,
    combine_instructionv2,
    intents_dict,
    turn_generation,
)
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM


class ConversationGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        # Check if GPU is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intents_dict = intents_dict
        # Initialize the model and tokenizer here as before
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            truncation=True,
            padding=True,
            padding_side="left",
            maximum_length=2048,
            model_max_length=2048,
        )

        # Determine model loading settings based on device availability
        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_4bit=True, device_map="auto"
            )
        else:
            # CPU only, load model without 4-bit quantization and manual device setting
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

    def generate_dialogue(
        self, topic, topic_type, context, list_of_intents, first_question
    ):
        combined_instructions = {}
        utterances_with_intents = []
        list_of_intents_original = list(list_of_intents)
        conv_turns = get_turn(list_of_intents)
        message = (
            "I will give you an entity, its type, and a background document, along with the user's first question to start a QA dialogue."
            + self.intents_dict[list_of_intents[0]]["user instruction"]
            + "\n"
            + topic
            + "\n"
            + context
            + "\n"
            + self.intents_dict[list_of_intents[0]]["user generation"]
            + first_question
        )
        str_output = message
        str_output = filter_new_turn(str_output, message, ":", ":")
        str_output = trim_to_last_punctuation(str_output)
        key = ":".join(str_output.strip().split("\n")[-1].split(":")[1:]).strip()
        utterances_with_intents.append({key: "OQ"})
        for i in range(1, len(list_of_intents)):
            turn = "user" if conv_turns[i] == 0 else "agent"
            j = 1
            while True:
                try:
                    previous_reply = self.intents_dict[list_of_intents[i - j]][
                        turn + " generation"
                    ].replace(":", ".")
                    break
                except KeyError:
                    j += 1
            if "_" in list_of_intents[i]:
                instruction_to_message = combine_instructionv2(
                    list_of_intents[i],
                    self.intents_dict,
                    turn + " instruction",
                    self.model,
                    self.tokenizer,
                )
                if list_of_intents[i] not in combined_instructions:
                    combined_instructions[list_of_intents[i]] = {}
                combined_instructions[list_of_intents[i]][
                    turn + " instruction"
                ] = instruction_to_message
                generation_to_message = self.intents_dict[list_of_intents[i].split("_")[0]][
                    turn + " generation"
                ]
            else:
                instruction_to_message = self.intents_dict[list_of_intents[i]][
                    turn + " instruction"
                ]
                generation_to_message = self.intents_dict[list_of_intents[i]][
                    turn + " generation"
                ]
            conversation_history = "\n".join(str_output.strip().split("\n"))
            message = (
                "I will give you an entity, its type, and a background document, and a conversation history that ends in a "
                + previous_reply
                + " "
                + instruction_to_message
                + "\n"
                + conversation_history
                + "\n"
                + generation_to_message
            )
            str_output = turn_generation(message, self.model, self.tokenizer)
            str_output = filter_new_turn(str_output, message, ":", ":")
            str_output = trim_to_last_punctuation(str_output)
            key = ":".join(str_output.strip().split("\n")[-1].split(":")[1:]).strip()
            utterances_with_intents.append({key: list_of_intents_original[i]})
        return {
            "generated_dialogue": utterances_with_intents,
            "entity_card_obj": None,
        }, combined_instructions

    def __repr__(self):
        return f"<ConversationGenerator using model {self.model_name}>"

    def set_intents_dict(self, intents_dict):
        self.intents_dict = intents_dict
