from collections import defaultdict, Counter
from functools import reduce
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

import regex as re


class multiTokenizer: 
    def __init__(self):
        self.queue = []

    def give_job(): 
        pass

class Tokenizer: 
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]): 
        self.input_path = input_path
        self.vocab_size = vocab_size

        self.jobs: list[tuple(str, str)] = []

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


def pretokenization(text: str, special_token: bytes) -> dict[bytes, int]:
    vocab = {} 
    vocab_count = 0
    paragraphs: list[str] = [] 

    assert isinstance(special_token, bytes)
    # special_token_re = [re.escape(token) for token in special_tokens] 
    special_token = special_token.decode("utf-8")
    special_token_re = re.escape(special_token)
    # special_token_re = "|".join(special_token_re.decode("utf-8"))
    special_token_re = fr"({special_token_re})".encode("utf-8")
    print(special_token_re)

    assert isinstance(special_token_re, bytes)

    text = text.encode("utf-8")
    assert isinstance(text, bytes)

    PAT = r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 

    text_start = 0 
    text_end = -1 
    for match in re.finditer(special_token_re, text): 
        paragraph = match.group(0) 
        paragraph = paragraph.strip()
        
        # Update the end with the start of the special token
        # and after appending to paragraph update the start with the end 
        # of the special token
        print(match.start())
        print(match.end())
        text_end = match.start() 
        paragraphs.append(text[text_start:text_end])
        text_start = match.end()

    for paragraph in paragraphs:
        for match in re.finditer(PAT.encode("utf-8"), paragraph):
            word = match.group()
            # print(f"Match: {word}")
            word = word.strip()
            word = word.decode()

            assert isinstance(word, str)

            word = ",".join(word)
            word = word.encode("utf-8") 
            # print(f"Word Type: {type(word)}")

            # self.vocab[word] = self.vocab_count
            if word in vocab: 
                vocab[word] += 1 
            else:
                vocab[word] = 1
            # vocab_count += 1
    
    return vocab

def callback(arg): 
    print("Hello from process")

# Creating function to make it easier to test
def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]): 
    vocab = {}
    freq_table = {}
    file = open(input_path, "rb")
    num_process = 6 
    vocab_count = 0

    # special_tokens_byte = [word.decode("utf-8") for word in special_tokens]

    boundaries = find_chunk_boundaries(file, num_process, special_tokens[0])
    running_process = []

    with Pool(processes=4) as pool:
        for start, end in zip(boundaries[:-1], boundaries[1:]): 
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

            process = pool.apply_async(
                pretokenization, (chunk, special_tokens[0],))
            running_process.append(process)

        for i in range(len(running_process)): 
            freq_table = Counter(vocab) + Counter(running_process[i].get())
        # vocab = reduce(lambda a, b: Counter(a.get()) + Counter(b), running_process)
    
    merges = []

    print("Pre-Tokenization")
    print(len(freq_table))

    # Tokenization - Begin BPE process
    pairs = defaultdict(int) 
    while len(vocab) < vocab_size:

        for (word, count) in freq_table.items(): 
            if word == b"<|endoftext|>" or word == b"" or len(word) < 1:
                continue

            print(f"Vocab: {word}")
            print(f"Vocab type: {type(word)}")
            letters = word.split(b",")
            for i in range(1, len(letters)): 
                pair = letters[i - 1: i + 1]
                pair = b",".join(pair)

                print(f"Pair: {pair}")
                # Double check this area
                pairs[pair] = count

        pair_combined = max(pairs, key=pairs.get)  
        print(f"Pair combined: {pair_combined}")

        new_key = pair_combined.replace(b",", b"") 
        old_key = pair_combined

        pair = (pair_combined.split(b",")[0], pair_combined.split(b",")[1])
        merges.append(pair)

        print(f"old_key: {old_key}")
        print(f"new_key: {new_key}")

        print(f"old key type: {type(old_key)}")

        new_words = {}
        for (word, count) in freq_table.items():   
            print(f"Old Key: {old_key}")
            print(f"vocab bytes: {word}")
            print(f"count: {count}")

            if old_key in word:      
                new_word = word.replace(old_key, new_key)
                print(f"replaced:  {new_word}")
                new_words[new_word] = word 

                # Update the freq table as well


        vocab_count += 1
        vocab[new_key] = vocab_count 
        # self.vocab[new_key] = self.vocab_count

        print("New Words")
        print(new_words)

        for (new_word, old_word) in new_words.items():
            print(f"New Vocab count: {new_word}")
            count = freq_table[old_word]
            freq_table[new_word] = count 

            del freq_table[old_word]

        pairs.clear() 


        print("Vocab: ")
        print(vocab)

        print("Merges: ")
        print(merges)
        


if __name__ == '__main__': 
    # bpe = Tokenizer("test", 10, ["<|endoftext|>"]) 
    encodings = run_train_bpe(
        input_path="test_2.text",
        vocab_size=100,
        special_tokens=[b"<|endoftext|>"]
    )
    print(f"Encodings: {encodings}")