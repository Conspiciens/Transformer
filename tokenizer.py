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
    def __init__(self, input_path: str): 
        self.input_path = input_path

        self.vocab = defaultdict(int) 
        self.vocab[b"<|endoftext|>"] = 0 
        self.vocab_count = 1 

        for i in range(97,123): 
            letter = chr(i)

            self.vocab[letter.encode()] = self.vocab_count
            self.vocab_count += 1 

        self.merges: list[tuple[bytes, ...]] = []
        self.special_tokens: list[str] = ["<|endoftext|>", "<|lost|>"]

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)


    def from_files(cls, vocab_filepth, merges_filepth, special_tokens=None): 
        pass

    def encode(self, text: str) -> list[int]: 
        special_token_re = [re.escape(token) for token in self.special_tokens] 
        special_token_re = "|".join(special_token_re)
        special_token_re = fr"({special_token_re})"

        print(special_token_re)
        
        PAT = r"""(<\|endoftext\|>)|(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 

        paragraphs = []
        text_start = 0 
        text_end = -1 
        # Remove all special tokens
        for match in re.finditer(special_token_re, text): 
            paragraph = match.group(0) 
            paragraph = paragraph.strip()
            
            # Update the end with the start of the special token
            # and after appending to paragraph update the start with the end 
            # of the special token
            text_end = match.start() 
            paragraphs.append(text[text_start:text_end])
            text_start = match.end()

        # Pre-tokenization - count all the words 
        for paragraph in paragraphs:
            for match in re.finditer(PAT, paragraph):
                word = match.group()
                print(f"Match: {word}")
                word = word.strip()

                print(f"Word: {word}")

                word = ",".join(word)
                word = word.encode() 
                print(f"Word Type: {type(word)}")

                self.vocab[word] = self.vocab_count
                self.vocab_count += 1

        print("Pre-Tokenization")
        print(self.vocab)

        # Tokenization - Begin BPE process
        self.pairs = defaultdict(int) 
        for i in range(6):

            for (vocab, count) in self.vocab.items(): 
                if vocab == b"<|endoftext|>" or vocab == "" or len(vocab) <= 1:
                    continue

                print(f"Vocab: {vocab}")
                print(f"Vocab type: {type(vocab)}")
                letters = vocab.split(b",")
                for i in range(1, len(letters)): 
                    pair = letters[i - 1: i + 1]
                    pair = b",".join(pair)

                    print(f"Pair: {pair}")
                    # Double check this area
                    self.pairs[pair] = count

            pair_combined = max(self.pairs, key=self.pairs.get)  
            print(f"Pair combined: {pair_combined}")

            new_key = pair_combined.replace(b",", b"") 
            old_key = pair_combined

            pair = (pair_combined.split(b",")[0], pair_combined.split(b",")[1])
            self.merges.append(pair)

            print(f"old_key: {old_key}")
            print(f"new_key: {new_key}")

            print(f"old key type: {type(old_key)}")

            new_words = {}
            for (vocab, count) in self.vocab.items():   
                print(f"Old Key: {old_key}")
                print(f"vocab bytes: {vocab}")

                if old_key in vocab:      
                    old_word = vocab
                    new_word = vocab.replace(old_key, new_key)
                    print(f"replaced:  {new_word}")
                    new_words[old_word] = new_word 

            self.vocab_count += 1
            self.vocab[new_key] = self.vocab_count

            print("New Words")
            print(new_words)

            for (old_word, new_word) in new_words.items():
                self.vocab[new_word] = self.vocab[old_word]
                del self.vocab[old_word]

            self.pairs.clear() 

            print("Vocab: ")
            print(self.vocab)

            print("Merges: ")
            print(self.merges)


    def encode_iterable(): 
        pass

    def decode(): 
        pass 
