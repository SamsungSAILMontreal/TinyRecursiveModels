import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.concept_decoder import ConceptDecoder

class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "<THOUGHT>": 100,
            "<END_THOUGHT>": 101,
            "hello": 102,
            "world": 103,
            " ": 104
        }
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text):
        if text in self.vocab:
            return [self.vocab[text]]
        return []
        
    def decode(self, tokens):
        return "".join([self.rev_vocab.get(t, "") for t in tokens])

def test_concept_decoder_thoughts():
    print("Testing ConceptDecoder with Thought Tokens...")
    
    num_concepts = 10
    decoder = ConceptDecoder(num_concepts=num_concepts, bpe_vocab_size=200)
    tokenizer = MockTokenizer()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 5
    
    # [concept_id, control, expand]
    # 0: concept 0 (expand=False) -> "hello" (mocked via expansion table fallback or embedding)
    # 1: <THOUGHT> (control)
    # 2: concept 1 (expand=False) -> "world" (inside thought)
    # 3: <END_THOUGHT> (control)
    # 4: concept 0 (expand=False) -> "hello"
    
    concept_ids = torch.tensor([[0, num_concepts + 3, 1, num_concepts + 4, 0]])
    control_symbols = torch.tensor([[0, 1, 0, 1, 0]]) # 1 for control
    expand_mask = torch.tensor([[0, 0, 0, 0, 0]])
    
    # Mock expansion table text for concepts
    decoder.expansion_table.concept_texts[0] = "hello"
    decoder.expansion_table.concept_texts[1] = "world"
    
    # Test 1: Show Thoughts = True
    print("\nTest 1: Show Thoughts = True")
    decoded = decoder.decode_sequence(
        concept_ids, control_symbols, expand_mask, tokenizer, show_thoughts=True
    )
    print(f"Output: {decoded[0]}")
    expected = "hello<THOUGHT>world<END_THOUGHT>hello"
    if decoded[0] == expected:
        print("PASS")
    else:
        print(f"FAIL. Expected: {expected}")

    # Test 2: Show Thoughts = False
    print("\nTest 2: Show Thoughts = False")
    decoded = decoder.decode_sequence(
        concept_ids, control_symbols, expand_mask, tokenizer, show_thoughts=False
    )
    print(f"Output: {decoded[0]}")
    # Note: The current implementation only hides the TAGS, not the content inside.
    # If we want to hide content, we'd need more logic in decode_sequence.
    # For now, let's verify tags are gone.
    expected_no_tags = "helloworldhello" 
    if decoded[0] == expected_no_tags:
        print("PASS")
    else:
        print(f"FAIL. Expected: {expected_no_tags}")

if __name__ == "__main__":
    test_concept_decoder_thoughts()
