import unittest
from solid_conversation import ConversationGenerator

class TestConversationGenerator(unittest.TestCase):
    
    def setUp(self):
        self.llm_model_name = "openai-community/gpt2"
        self.generator = ConversationGenerator(self.llm_model_name)  # Initialize the class 

        
    def test_generate_dialogue(self):
        entity_name = "Jönköping"
        entity_type = "City"
        entity_description = ("Jönköping (Swedish pronunciation: [jœnˈøːnpɪŋ] (listen)) is a city in southern Sweden, situated by the western shore of Lake Vättern. "
                              "It is the seat of Jönköping municipality and Jämtland County, and has a population of 114,418 (2019). Jönköping is part of the Swedish province "
                              "of Småland. Historically, the city has been significant due to its location at the transition between the provinces of Västergötland and Småland, "
                              "which is reflected in its architecture and cultural heritage.")
        sequence_of_intents = ['OQ', 'RQ', 'FD_NF', 'PA']
        conversation_starter = "How has the cultural and architectural heritage of both Västergötland and Småland influenced the development of Jönköping as a unique city?"
        
        solid_generated_dialogue, multi_intent_self_instructions = self.generator.generate_dialogue(
            entity_name, entity_type, entity_description, sequence_of_intents, conversation_starter
        )
        
        self.assertIn('generated_dialogue', solid_generated_dialogue)
        self.assertIsInstance(solid_generated_dialogue['generated_dialogue'], list)
        self.assertGreater(len(solid_generated_dialogue['generated_dialogue']), 0)

if __name__ == "__main__":
    unittest.main()
