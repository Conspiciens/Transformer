from collections import defaultdict, Counter
from functools import reduce
import pathlib
from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

import heapq
import regex as re
import cProfile, pstats


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


def pretokenization(text: str, special_token: list[str]) -> dict[bytes, int]:
    vocab = {} 

    paragraphs: list[str] = [] 

    # special_token_re = [re.escape(token) for token in special_tokens] 
    # special_token = special_token.encode("utf-8")
    # special_token_re = re.escape(special_token)
    # special_token_re = "|".join(special_token_re.decode("utf-8"))
    # special_token_re = fr"({special_token_re})".encode("utf-8")
    special_token_re = fr"(<|endoftext|>)"# .encode("utf-8")
    # print(special_token_re)

    # assert isinstance(special_token_re, bytes)

    # text = text.encode("utf-8")
    # assert isinstance(text, bytes)

    # PAT = r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    text_start = 0 
    text_end = -1 
    for match in re.finditer(special_token_re, text): 
        paragraph = match.group(0) 
        
        # Update the end with the start of the special token
        # and after appending to paragraph update the start with the end 
        # of the special token
        text_end = match.start() 
        paragraphs.append(text[text_start:text_end])
        text_start = match.end()


    # Found myself intrigued by the fact that it 
    # required text to be a string, and couldn't compare with 
    # bytes requires a little more research then...     
    paragraphs.append(text[text_start:])

    for paragraph in paragraphs:
        for match in re.finditer(PAT, paragraph):
            word = match.group()
            word = word.encode("utf-8")
            word_tuple = tuple(bytes([b]) for b in word)

            if word_tuple in vocab: 
                vocab[word_tuple] += 1 
            else:
                vocab[word_tuple] = 1
    
    return vocab

def callback(arg): 
    print("Hello from process")

# Creating function to make it easier to test
def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]): 
    vocab = {}
    vocab[0] = b"<|endoftext|>"
    vocab_count = 1

    for i in range(256): 
        letter = chr(i)

        # Change this 
        vocab[vocab_count] = bytes([i])
        # vocab[bytes([i])] = vocab_count 
        vocab_count += 1


    freq_table = Counter()
    file = open(input_path, "rb")
    num_process = 6 

    # special_tokens_byte = [word.decode("utf-8") for word in special_tokens]

    boundaries = find_chunk_boundaries(file, num_process, special_tokens[0].encode("utf-8"))
    running_process = []

    with Pool(processes=6) as pool:
        for start, end in zip(boundaries[:-1], boundaries[1:]): 
            # Example code take from pretokenization example 
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

            # process = pool.apply_async(
            #     pretokenization, (chunk, special_tokens[0],))


            # Using .get() now over the over loop saved us about .1 seconds
            running_process.append((chunk, special_tokens[0],)) 
            # i += 1
        
        processes = pool.starmap(pretokenization, running_process)
        freq_table = reduce(lambda a, b: Counter(a) + Counter(b), processes)
    
    merges = [None] * (vocab_size - len(vocab)) 
    y = 0

    # merges = []

    pairs = defaultdict(int) 
    for (word, count) in freq_table.items(): 
        for i in range(1, len(word)): 
            pair = (word[i - 1], word[i])
            pairs[pair] += count


    # pairs = heapq.heapify(pairs)

    words_changed = {}
    words_freq = {}

    # Tokenization - Begin BPE process
    while len(vocab) < vocab_size:
        # pairs = defaultdict(int) 

        # for (word, count) in freq_table.items(): 
        #     for i in range(1, len(word)): 
        #         pair = (word[i - 1], word[i])
        #         pairs[pair] += count

        pair_combined = max(pairs, key=lambda k: (pairs[k], k))  
        print(f"Max: {pair_combined}")
        prefix, suffix = pair_combined

        old_key = (prefix, suffix)
        new_key = prefix + suffix
        
        merges[y] = old_key 
        y += 1

        vocab[vocab_count] = new_key
        vocab_count += 1

        # Updates the frequency table with the new pair and it's following count 
        new_freq_table = defaultdict(int)
        for (word, count) in freq_table.items():   
            word_len = len(word)

            result = [None] * word_len 
            idxs = []

            i = 0 
            word_count = 0
            pop_count = 0
            prev_idx = 0

            while word_count < word_len: 
                if word_count < word_len - 1 and old_key[0] == word[word_count] and old_key[1] == word[word_count + 1]:
                    result[i] = new_key

                    # Should append when there's an idx that changes
                    idxs.append(word_count)

                    i += 1
                    word_count += 2
                    pop_count += 1
                else: 
                    result[i] = word[word_count]
                    word_count += 1
                    i += 1

            # Ok, it's better to just recount at the end (bruhhhh so easy)
            if pop_count > 0: 
                del result[(pop_count * -1):]

                for i in range(1, len(word)): 
                    pair = (word[i - 1], word[i])
                    pairs[pair] -= count

                for i in range(1, len(result)): 
                    pair = (result[i - 1], result[i])
                    pairs[pair] += count


            new_freq_table[tuple(result)] += count

            if len(idxs) <= 0: 
                continue 
            words_changed[word] = idxs

        freq_table = new_freq_table
        del pairs[old_key]
        
    return (vocab, merges)


if __name__ == '__main__': 
    # bpe = Tokenizer("test", 10, ["<|endoftext|>"]) 
    profile = cProfile.Profile()
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "tests" / "fixtures"

    input_path = FIXTURES_PATH / "corpus.en"
    profile.enable()
    (vocab, merges) = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    profile.disable()

    stats = pstats.Stats(profile).sort_stats('cumulative')
    stats.print_stats(30)
