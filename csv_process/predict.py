from transformers import  TFGPT2LMHeadModel
import pickle
import logging


class TextPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """Loads the trained model and tokenizer"""
        try:
            # Load the tokenizer
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logging.info("Tokenizer successfully loaded")

            # Load the model
            self.model = TFGPT2LMHeadModel.from_pretrained(self.model_path)
            logging.info("Model successfully loaded")

        except Exception as e:
            logging.error(f"Model/Tokenizer loading error: {str(e)}")
            raise

    def generate_text(self, prompt, max_length=50, temperature=0.7):
        """Generates text based on the given prompt"""
        try:
            # Tokenize the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='tf')

            # Generate text
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode tokens to text
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logging.error(f"Text generation error: {str(e)}")
            return prompt


def main():
    # Model and tokenizer paths
    model_path = "models/model_TIMESTAMP/best_model.keras"  # Replace TIMESTAMP with actual value
    tokenizer_path = "tokenizer.pickle"

    try:
        # Initialize the text predictor
        predictor = TextPredictor(model_path, tokenizer_path)

        # Test prompts
        test_prompts = [
            "fun main() {",
            "class MyClass {",
            "val result = ",
            "fun calculate("
        ]

        # Generate predictions for each prompt
        for prompt in test_prompts:
            generated = predictor.generate_text(
                prompt,
                max_length=50,
                temperature=0.7
            )
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {generated}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
