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

        self.vocab = defaultdict(int) 
        self.vocab[0] = b"<|endoftext|>" 
        self.vocab_count = 1 

        for i in range(97,123): 
            letter = chr(i)

            self.vocab[self.vocab_count] = letter.encode()
            self.vocab_count += 1 

        self.merges: list[tuple[bytes, ...]] = []
        # self.special_tokens: list[str] = ["<|endoftext|>", "<|lost|>"]
        self.special_tokens: list[str] = special_tokens

        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)


    def from_files(cls, vocab_filepth, merges_filepth, special_tokens=None): 
        pass

    def encode(self, text: str) -> list[int]: 
        pass

    def encode_iterable(): 
        pass

    def decode(): 
        pass 


def pretokenization(text: str, special_token: list[str]) -> dict[bytes, int]:
    vocab = {} 

    paragraphs: list[str] = [] 

    special_token_re = [re.escape(token) for token in special_token]
    special_token_re = "|".join(special_token_re)
    special_token_re = fr"({special_token_re})"

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


# Creating function to make it easier to test
def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]): 
    vocab = {}
    vocab_count = 0 
    
    special_tokens_bytes = [bytes(token, "utf-8") for token in special_tokens]

    # Add all unique tokens in the vocab = {}
    for i, token in enumerate(special_tokens_bytes):
        vocab[i] = token
        vocab_count += 1

    # Add the alphabet to the range
    for i in range(256): 
        letter = chr(i)

        # Change this 
        vocab[vocab_count] = bytes([i])
        # vocab[bytes([i])] = vocab_count 
        vocab_count += 1


    freq_table = Counter()
    file = open(input_path, "rb")
    num_process = 6 

    boundaries = find_chunk_boundaries(file, num_process, special_tokens_bytes)
    running_process = []

    #TODO: Any performance improvements we can improve here? 
    with Pool(processes=6) as pool:
        for start, end in zip(boundaries[:-1], boundaries[1:]): 
            # Example code take from pretokenization example 
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

            # Using .get() now over the over loop saved us about .1 seconds
            running_process.append((chunk, special_tokens)) 

        processes = pool.starmap(pretokenization, running_process)
        freq_table = reduce(lambda a, b: Counter(a) + Counter(b), processes)
    
    merges = [None] * (vocab_size - len(vocab)) 
    y = 0

    pairs = defaultdict(int) 
    for (word, count) in freq_table.items(): 
        for i in range(1, len(word)): 
            pair = (word[i - 1], word[i])
            pairs[pair] += count


    # Tokenization - Begin BPE process
    while len(vocab) < vocab_size:

        pair_combined = max(pairs, key=lambda k: (pairs[k], k))  
        # print(f"Max: {pair_combined}")
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

            i = 0 
            word_count = 0
            pop_count = 0
            prev_idx = 0

            while word_count < word_len: 
                if word_count < word_len - 1 and old_key[0] == word[word_count] and old_key[1] == word[word_count + 1]:
                    result[i] = new_key


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
