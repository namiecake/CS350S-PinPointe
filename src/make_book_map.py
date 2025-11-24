#!/usr/bin/env python3
import json
from pathlib import Path

def load_json(filepath):
    """Load a JSON file."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    datapath = data_dir / filepath
    with open(datapath, 'r') as f:
        return json.load(f)

def load_jsonl(filepath):
    """Load a JSONL file (one JSON object per line)."""
    books = {}
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    datapath = data_dir / filepath
    with open(datapath, 'r') as f:
        for line in f:
            book = json.loads(line.strip())
            books[str(book['book_id'])] = book['title']
    return books

def create_book_index_to_title_mapping(book_id_to_index, book_id_to_title):
    """Create a mapping from book index to book title."""
    print("\nCreating book index to title mapping...")
    
    book_index_to_title = {}
    missing_titles = 0
    
    for book_id, index in book_id_to_index.items():
        if book_id in book_id_to_title:
            book_index_to_title[index] = book_id_to_title[book_id]
        else:
            # If title not found, use a placeholder
            book_index_to_title[index] = f"Unknown (ID: {book_id})"
            missing_titles += 1
    
    print(f"  Created mapping for {len(book_index_to_title)} books")
    if missing_titles > 0:
        print(f"  Warning: {missing_titles} books had missing titles")
    
    return book_index_to_title

def save_book_index_to_title_mapping(book_index_to_title, output_file):
    """Save book index to title mapping to a JSON file."""
    print(f"\nSaving book index to title mapping to {output_file}...")
    
    mapping_data = {
        'metadata': {
            'n_books': len(book_index_to_title),
            'description': 'Mapping from book vector index to book title'
        },
        'book_index_to_title': book_index_to_title
    }
    
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"  Successfully saved mapping for {len(book_index_to_title)} books")

def main():  
    # Load all data files
    print("Loading data files...")
    book_id_to_title = load_jsonl('goodreads_books_children.json')
    book_id_to_index_data = load_json('book_id_to_index.json')
    book_id_to_index = book_id_to_index_data['book_id_to_index']
    
    print(f"  Loaded {len(book_id_to_title)} book titles")
    print(f"  Loaded {len(book_id_to_index)} book indices")
    
    # Create the index to title mapping
    book_index_to_title = create_book_index_to_title_mapping(book_id_to_index, book_id_to_title)
    
    # Save to output file
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    output_path = data_dir / 'book_index_to_title.json'
    
    save_book_index_to_title_mapping(book_index_to_title, output_path)
    
    print("\n✓ Done! Book index to title mapping created successfully.")

if __name__ == "__main__":
    main()