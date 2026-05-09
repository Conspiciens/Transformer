from collections import defaultdict
from pretokenization_example import find_chunk_boundaries

import regex as re
import multiprocessing


class multiTokenizer: 
    def __init__(self):
        self.queue = []

    def give_job(): 
        pass

class Tokenizer: 
    def __init__(self): 
        self.input_path = "TinyStoriesV2-GPT4-valid.txt"

        self.vocab = defaultdict(int) 
        self.vocab[b"<|endoftext|>"] = 0
        self.vocab_count = 1 

        for i in range(97,123): 
            letter = chr(i)

            self.vocab[letter.encode('utf-8')] = self.vocab_count 
            self.vocab_count += 1 

        self.merges: list[tuple[bytes, ...]] = []
        self.special_tokens: list[str] = ["<|endoftext|>"]

    def from_files(cls, vocab_filepth, merges_filepth, special_tokens=None): 
        pass

    def encode(self, text: str) -> list[int]: 
        PAT = r"""(<\|endoftext\|>)|(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 

        print(f"Input: {text}")        

        # Pre-tokenization - count all the words 
        for match in re.finditer(PAT, text):
            word = match.group()
            print(f"Match: {word}")
            word = word.strip()

            print(f"Word: {word}")
            
            # Probably later down the line we need to handle this
            if word == "<|endoftext|>":
                continue

            word = ",".join(word)
            word = word.encode() 

            self.vocab[word] += 1

        print("Pre-Tokenization")
        print(self.vocab)

        # Tokenization - Begin BPE process
        self.pairs = defaultdict(int) 
        for i in range(6):

            for (vocab, count) in self.vocab.items(): 
                # Temp fix
                if vocab == b"<|endoftext|>" or vocab == b"":
                    continue

                print(f"Word: {vocab}")
                letters = vocab.split(b",")
                for i in range(1, len(letters)): 
                    pair = letters[i - 1: i + 1]
                    pair = b",".join(pair)

                    print(f"Pair: {pair}")
                    self.pairs[pair] += count

            pair_combined = max(self.pairs, key=self.pairs.get)  
            print(f"Pair combined: {pair_combined}")

            new_key = pair_combined.replace(b",", b"") 
            old_key = pair_combined

            pair = (pair_combined.split(b",")[0], pair_combined.split(b",")[1])
            self.merges.append(pair)

            print(f"old_key: {old_key}")
            print(f"new_key: {new_key}")

            new_words = {}
            for (vocab, count) in self.vocab.items():   
                # Temp fix
                # vocab = str(vocab)
                if old_key in vocab:         
                    old_word = vocab
                    new_word = vocab.replace(old_key, new_key)
                    print(f"replaced:  {new_word}")
                    new_words[old_word] = new_word 

            print("New Words")
            print(new_words)

            for (old_word, new_word) in new_words.items(): 
                self.vocab[new_word] = self.vocab[old_word]
                del self.vocab[old_word]

            self.vocab_count += 1
            self.pairs.clear() 

            print("Vocab: ")
            print(self.vocab)

            print("Merges: ")
            print(self.merges)


    def encode_iterable(): 
        pass

    def decode(): 
        pass 
