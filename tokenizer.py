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
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]): 
        self.input_path = input_path
        self.vocab_size = vocab_size

        # self.vocab: dict[int, bytes]
        self.vocab = defaultdict(int) 
        self.vocab[0] = b"<|endoftext|>" 
        self.vocab_count = 1 

        for i in range(97,123): 
            letter = chr(i)

            self.vocab[self.vocab_count] = letter.encode()
            # self.vocab[letter.encode()] = self.vocab_count
            self.vocab_count += 1 

        self.merges: list[tuple[bytes, ...]] = []
        # self.special_tokens: list[str] = ["<|endoftext|>", "<|lost|>"]
        self.special_tokens: list[str] = special_tokens

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

                # self.vocab[word] = self.vocab_count
                self.vocab[self.vocab_count] = word
                self.vocab_count += 1

        print("Pre-Tokenization")
        print(self.vocab)

        # Tokenization - Begin BPE process
        self.pairs = defaultdict(int) 
        while len(self.vocab) < self.vocab_size:

            for (count, vocab) in self.vocab.items(): 
                if vocab == b"<|endoftext|>" or vocab == b"" or len(vocab) < 1:
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
            for (count, vocab) in self.vocab.items():   
                print(f"Old Key: {old_key}")
                print(f"vocab bytes: {vocab}")
                print(f"count: {count}")

                if old_key in vocab:      
                    new_word = vocab.replace(old_key, new_key)
                    print(f"replaced:  {new_word}")
                    new_words[count] = new_word 

            self.vocab_count += 1
            self.vocab[self.vocab_count] = new_key
            # self.vocab[new_key] = self.vocab_count


            print("New Words")
            print(new_words)

            for (count, new_word) in new_words.items():
                print(f"Old Vocab count: {count}")
                print(f"New Vocab count: {new_word}")
                self.vocab[count] = new_word 
                # self.vocab[new_word] = self.vocab[old_word]

            self.pairs.clear() 

            print("Vocab: ")
            print(self.vocab)

            print("Merges: ")
            print(self.merges)


    def encode_iterable(): 
        pass

    def decode(): 
        pass 


if __name__ == '__main__': 
    bpe = Tokenizer("test", 10, ["<|endoftext|>"]) 
    encodings = bpe.encode('''u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>''')
    print(f"Encodings: {encodings}")