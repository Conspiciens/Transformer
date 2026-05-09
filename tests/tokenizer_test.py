import pytest 
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer import Tokenizer 

def test_init_tokenizer(): 
    bpe = Tokenizer() 
    
    print(bpe.vocab)

def test_encoding_phrase(): 
    bpe = Tokenizer() 
    
    encodings = bpe.encode('''low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest''')
    print(f"Encodings: {encodings}")

    # answer = ["st", "est", "ow", "low", "west", "ne"]

    # assert encodings == answer

def test_paragraph(): 
    bpe = Tokenizer() 
    encodings = bpe.encode('''u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>''')
    print(f"Encodings: {encodings}")

def test_encoding_document(): 
    pass 
