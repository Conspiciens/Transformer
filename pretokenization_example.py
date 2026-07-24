import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, list), "Must represent special token as a bytestring"
    print(split_special_token)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess

        # Read a mini chunk size, however we want to 
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            special_token_idx: list[int] = [] 
            # Loop through the special tokens to search for within a chunk
            for token in split_special_token: 

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(token)
                if found_at != -1:
                    special_token_idx.append(initial_position + found_at)

                
            # Check how many special tokens are found
            # if more than 1 are found then find the 
            # closest to the intial_position
            if len(special_token_idx) == 1: 
                chunk_boundaries[bi] = initial_position + found_at
                break 
            elif len(special_token_idx) > 1:
                print("Looping")
                # Searching which special_token comes first, whichever comes 
                # first is assigned the chunk boundary 
                chunk_boundaries[bi] = special_token_idx[1]
                for idx in range(1, len(special_token_idx)):
                    if special_token_idx[idx - 1] < special_token_idx[idx]:
                        chunk_boundaries[bi] = special_token_idx[idx - 1]
                        print(chunk_boundaries[bi])
                break 


            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
# 
#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token