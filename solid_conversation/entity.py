import os
import string
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class EntityGenerator:
    def __init__(self, model_name, base_out_path, letter):
        self.device = torch.device('cuda')
        self.model_name = model_name
        self.base_out_path = base_out_path
        self.letter = letter
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, padding_side="left", maximum_length=2048, model_max_length=2048)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.entity_types = self._get_entity_types()

    def _get_entity_types(self):
        return [
            "Person", "Organization", "Place", "Country", "City", "Product", "Service", 
            "Company", "Animal", "Plant", "Food", "Beverage", "Vehicle", "Book", "Movie", 
            "Song", "Artist", "Actor", "Actress", "Author", "Musician", "Athlete", 
            "Politician", "Celebrity", "Brand", "University", "School", "Hospital", 
            "Government Agency", "Nonprofit Organization", "Event", "Conference", 
            "Festival", "Sport", "Team", "League", "Game", "Software", "App", 
            "Website", "Social Media Platform", "Technology", "Device", "Gadget", 
            "Instrument", "Tool", "Furniture", "Clothing", "Fashion Brand", "Artwork", 
            "Painting", "Sculpture", "Architectural Structure", "Historical Figure", 
            "Mythical Creature", "Deity", "Supernatural Being", "Character (Fictional)", 
            "Language", "Programming Language", "Genre (Music, Film, Literature)", 
            "Style (Fashion, Art)", "Historical Period", "Scientific Concept", 
            "Chemical Element", "Particle", "Planet", "Star", "Galaxy", "Constellation", 
            "Astronomical Object", "Natural Disaster", "Weather Phenomenon", "Disease", 
            "Medication", "Medical Procedure", "Law", "Legal Case", "Political Ideology", 
            "Social Movement", "Philosophy", "Religion", "Mythology", "Folklore", 
            "Cuisine", "Recipe", "Sportsperson", "Entrepreneur", "Inventor", 
            "Scientist", "Mathematician", "Philosopher", "Explorer", "Author", 
            "Poet", "Photographer", "Journalist", "Activist", "Historical Event"
        ]

    def _generate_entities(self, inputs, output_paths, batch_size):
        tokens = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        tokens = tokens.to(self.device)
        generated_queries = []

        for i in tqdm(range(0, len(tokens["input_ids"]), batch_size), desc="Generating entities..."):
            batch = tokens["input_ids"][i:i + batch_size]
            batch_output_paths = output_paths[i:i + batch_size]
            outputs = self.model.generate(
                input_ids=batch,
                attention_mask=tokens["attention_mask"][i:i + batch_size],
                min_new_tokens=1000,
                max_new_tokens=1000,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=2
            )
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for o, o_path in zip(outputs, batch_output_paths):
                o = o.strip()
                with open(o_path, "a+") as fp:
                    fp.write(str(o) + "\n")

        return True

    def run(self):
        prompt_template = """Instruction: Provide a list of at least 100 entities categorized as '{entity_type}' whose names begin with the letter '{letter}'. Please include a new line after each entity.
Entities:
"""
        inputs = []
        output_paths = []
        for entity_type in self.entity_types:
            output_path = self.base_out_path.format(entity_type, self.letter)
            if os.stat(output_path).st_size == 0:  # Only if it is empty
                output_paths.append(output_path)
                inputs.append(prompt_template.format(entity_type=entity_type, letter=self.letter))

        batch_size = 1  # Adjust batch size as needed

        for output_path in output_paths:
            open(output_path, "w+").close()  # Ensure the output file is created

        self._generate_entities(inputs, output_paths, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate entities based on a specific letter.")
    parser.add_argument("model_name", type=str, help="Model name from Hugging Face Hub")
    parser.add_argument("base_out_path", type=str, help="Base output path for generated entities")
    parser.add_argument("letter", type=str, help="The letter to filter generated entities")

    args = parser.parse_args()

    generator = EntityGenerator(args.model_name, args.base_out_path, args.letter)
    generator.run()
