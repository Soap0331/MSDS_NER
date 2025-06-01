import json
import re
from typing import List, Dict, Tuple, Any

def get_word_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Extract word spans from text.
    Returns list of (start, end, word) tuples.
    """
    # Using regex to find words (handling punctuation separately)
    words = []
    for match in re.finditer(r'\b\w+\b|[^\w\s]', text):
        start, end = match.span()
        word = match.group()
        words.append((start, end, word))
    return words

def get_bio_tags(text: str, labels: List[List[Any]]) -> List[Tuple[str, str]]:
    """
    Convert character-level spans to word-level BIO tags.
    Returns list of (word, bio_tag) tuples.
    """
    word_spans = get_word_spans(text)

    # Convert character-level spans to ranges for efficient lookup
    char_labels = {}
    for start, end, entity_type in labels:
        for i in range(start, end):
            char_labels[i] = entity_type

    # Assign BIO tags to each word
    bio_tags = []
    for word_start, word_end, word in word_spans:
        # Skip empty words (shouldn't happen, but just in case)
        if not word:
            continue

        # Check if any character in the word has a label
        word_chars_with_labels = [char_labels.get(i) for i in range(word_start, word_end)]

        # Filter out None values
        word_labels = [label for label in word_chars_with_labels if label]

        if not word_labels:
            # No label for this word
            bio_tags.append((word, "O"))
        else:
            # Get the most common label for this word (handling potential edge cases where a word has multiple labels)
            most_common_label = max(set(word_labels), key=word_labels.count)

            # Check if this word starts a new entity or continues one
            prev_char_idx = word_start - 1
            if prev_char_idx >= 0 and char_labels.get(prev_char_idx) == most_common_label:
                # Previous character had the same label, so this word continues an entity
                bio_tags.append((word, f"I-{most_common_label}"))
            else:
                # This word starts a new entity
                bio_tags.append((word, f"B-{most_common_label}"))

    return bio_tags

def process_jsonl_file(input_file: str, output_file: str):
    """
    Process the JSONL file and write BIO tagged output.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '')
                    labels = data.get('label', [])

                    bio_tags = get_bio_tags(text, labels)

                    # Write BIO tagged output
                    for word, tag in bio_tags:
                        outfile.write(f"{word}\t{tag}\n")

                    # Add a blank line between sequences
                    outfile.write("\n")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                except Exception as e:
                    print(f"Error processing line: {e}")

def demo_conversion():
    """
    Demonstrate conversion on a small example from the data.
    """
    sample_json = """{"id":7429,"text":"routes of entry: inhalation:yes skin:yes ingestion:yes health hazards acute and chronic:vapor can irritate the nose and throat.severly irritating to skin.severly irritating to eyes;possible permanent injury. effects of overexposure:may cause kidney and liver damage w\/prolonged exposure to xylene.overexposure to chlorinated hydrocarbons can cause headache,nervousness,nausea,and weakness,progressing to tremor and convulsions.repea ted exposure may cause liver damage.","meta":{"product_id":"KELTHANE(R) EC MITICIDE","msds_number":"BBBTD"},"label":[[98,106,"症状"],[111,115,"器官"],[120,126,"器官"],[127,145,"症状"],[149,153,"器官"],[154,172,"症状"],[176,180,"器官"],[190,206,"症状"],[242,248,"器官"],[253,258,"器官"],[259,265,"症状"],[348,356,"症状"],[357,368,"症状"],[369,375,"症状"],[380,388,"症状"],[404,410,"症状"],[415,426,"症状"],[456,461,"器官"],[462,468,"症状"]],"Comments":[]}"""

    data = json.loads(sample_json)
    text = data.get('text', '')
    labels = data.get('label', [])

    print("Original text:")
    print(text)
    print("\nLabels:")
    for start, end, label in labels:
        print(f"{start}-{end}: {text[start:end]} [{label}]")

    print("\nBIO tags:")
    bio_tags = get_bio_tags(text, labels)
    for word, tag in bio_tags:
        print(f"{word}\t{tag}")

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        process_jsonl_file(input_file, output_file)
        print(f"Converted {input_file} to BIO format in {output_file}")
    else:
        print("Usage: python convert_to_bio.py <input_jsonl_file> <output_bio_file>")
        print("\nRunning demo conversion instead...\n")
        demo_conversion()
        #python BIOlabel.py enhance0520.jsonl 0520_enhancedata.bio