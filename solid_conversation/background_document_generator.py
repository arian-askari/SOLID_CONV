import argparse
import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class BackgroundDocumentGenerator:
    def __init__(self, model_name, base_out_path, device):
        self.device = torch.device(device)
        self.model_name = model_name
        self.base_out_path = base_out_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, truncation=True, padding=True, padding_side="left", model_max_length=1024)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def _generate_background_documents(self, inputs, batch_size, output_paths):
        tokens = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        tokens = tokens.to(self.device)

        for i in tqdm.tqdm(range(0, len(tokens["input_ids"]), batch_size), desc="Generating background documents..."):
            batch = tokens["input_ids"][i:i + batch_size]
            batch_output_paths = output_paths[i:i + batch_size]
            outputs = self.model.generate(
                input_ids=batch,
                attention_mask=tokens["attention_mask"][i:i + batch_size],
                min_new_tokens=200, max_new_tokens=200,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=2
            )
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for o, o_path in zip(decoded_outputs, batch_output_paths):
                with open(o_path, "a+") as fp:
                    fp.write(o.strip())

    def generate_from_data(self, entities_parsed_path, batch_size):
        with open(entities_parsed_path, "r") as f:
            entities = json.load(f)

        entities = [(item.split("\t")[0], item.split("\t")[1]) for item in entities]

        prompt_template = """If you have no knowledge about the following entity, please write "N/A." 
        Otherwise, generate a background document based on Wikipedia for the specified entity.
        Entity type is "{entity_type}", and entity name is "{entity_name}".

        Background Document:"""

        inputs, output_paths = [], []

        for entity_name, entity_type in entities:
            output_path = self.base_out_path.format(
                entity_type, entity_name, self.model_name.replace("-", "_").replace("/", "_"))

            if not os.path.isfile(output_path) or os.stat(output_path).st_size == 0:
                output_paths.append(output_path)
                inputs.append(prompt_template.format(
                    entity_type=entity_type, entity_name=entity_name))
            else:
                print(f"Skipping existing file: {output_path}")

        print(f"Generating {len(inputs)} background documents...")

        # Pre-create output files
        for output_path in output_paths:
            open(output_path, "w+").close()

        self._generate_background_documents(inputs, batch_size, output_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate background documents for entities.")
    parser.add_argument("model_name", type=str, help="Pretrained model name (e.g., 'HuggingFaceH4/zephyr-7b-beta').")
    parser.add_argument("base_out_path", type=str, help="Base output path for storing generated documents.")
    parser.add_argument("entities_parsed_path", type=str, help="Path to the JSON file containing entity names and types.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on ('cuda' or 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()

    generator = BackgroundDocumentGenerator(
        model_name=args.model_name,
        base_out_path=args.base_out_path,
        device=args.device
    )

    generator.generate_from_data(
        entities_parsed_path=args.entities_parsed_path,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
