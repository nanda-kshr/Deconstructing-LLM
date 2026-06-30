import os
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class CustomTokenizer:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.tokenizer = HFTokenizer.from_file(model_path)
        else:
            self.tokenizer = None

    def train(self, corpus_path, vocab_size, save_path):
        """
        Trains a byte-level BPE tokenizer on the specified corpus.
        """
        self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "<|endoftext|>"]
        )
        
        # Train
        self.tokenizer.train([corpus_path], trainer=trainer)
        
        # Save
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self.tokenizer.save(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded or trained yet.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded or trained yet.")
        return self.tokenizer.decode(ids)
