import unittest
import CoreFunctionality as cf

class TestGUI(unittest.TestCase):
    
    def test_locate_file(self):
        # Correct path to Git repository
        input1 = "https://github.com/KnudRonau/design-pattern-showcasing"
        # Correct path to local LLM
        input2 = "D:/LmStudio/Models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q6_K.gguf"
        cf.setup(input1, input2, 0.5)
        self.assertIsNotNone(cf.llm)
        self.assertIsNotNone(cf.vector_database)
    

    def test_locate_file2(self):
        # Faulty path to Git repository
        input1 = "Faulty URL"
        # Correct path to local LLM
        input2 = "D:/LmStudio/Models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q6_K.gguf"
        cf.setup(input1, input2, 0.5)
        self.assertIsNone(cf.vector_database)
        self.assertIsNone(cf.llm)
    
    def test_locate_file3(self):
        # Correct path to Git repository
        input1 = "https://github.com/KnudRonau/design-pattern-showcasing"
        # Faulty path to local LLM
        input2 = "Faulty LLM path"
        cf.setup(input1, input2, 0.5)
        self.assertIsNone(cf.vector_database)
        self.assertIsNone(cf.llm)
    
    def test_locate_file4(self):
        # Faulty path to Git repository
        input1 = "Faulty URL"
        # Faulty path to local LLM
        input2 = "Faulty LLM path"
        cf.setup(input1, input2, 0.5)
        self.assertIsNone(cf.vector_database)
        self.assertIsNone(cf.llm)


if __name__ == '__main__':
    unittest.main()