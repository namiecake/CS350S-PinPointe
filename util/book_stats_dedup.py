#!/usr/bin/env python3
"""
Analyze Goodreads children's book reviews dataset
Provides statistics on users, books, reviews, and duplicates
Creates deduplicated versions of books and reviews datasets

paths might be wrong lol
"""

import json
from collections import Counter, defaultdict

def load_json_lines(filename):
    """Load JSON lines file into a list of dictionaries"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_json_lines(data, filename):
    """Save list of dictionaries to JSON lines file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data):,} entries to {filename}")

def main():
    # Load datasets
    print("Loading datasets...")
    reviews = load_json_lines('goodreads_reviews_children.json')
    books = load_json_lines('goodreads_books_children.json')
    
    print(f"Loaded {len(reviews):,} reviews")
    print(f"Loaded {len(books):,} books")
    print("=" * 80)
    
    # Build book_id to title mapping
    book_id_to_title = {book['book_id']: book['title'] for book in books}
    
    # Count reviews per user
    user_review_counts = Counter(review['user_id'] for review in reviews)
    
    # Count reviews per book
    book_review_counts = Counter(review['book_id'] for review in reviews)
    
    # User statistics
    print("\n" + "=" * 80)
    print("USER STATISTICS")
    print("=" * 80)
    print(f"Total unique users: {len(user_review_counts):,}")
    
    if user_review_counts:
        most_active_user, most_reviews = user_review_counts.most_common(1)[0]
        print(f"\nUser with most reviews: {most_active_user}")
        print(f"Number of reviews: {most_reviews:,}")
        
        least_active_user, least_reviews = user_review_counts.most_common()[-1]
        print(f"\nUser with least reviews: {least_active_user}")
        print(f"Number of reviews: {least_reviews:,}")
    
    # Book review statistics
    print("\n" + "=" * 80)
    print("BOOK REVIEW STATISTICS")
    print("=" * 80)
    
    books_with_reviews = len(book_review_counts)
    books_without_reviews = len(books) - books_with_reviews
    
    print(f"Books with reviews: {books_with_reviews:,}")
    print(f"Books without reviews: {books_without_reviews:,}")
    
    # Most reviewed books
    print("\n" + "=" * 80)
    print("TOP 20 MOST REVIEWED BOOKS")
    print("=" * 80)
    
    for i, (book_id, count) in enumerate(book_review_counts.most_common(20), 1):
        title = book_id_to_title.get(book_id, "Unknown Title")
        print(f"{i:2d}. {title[:70]}")
        print(f"    Book ID: {book_id}, Reviews: {count:,}")
    
    # Find most reviewed book
    if book_review_counts:
        most_reviewed_book_id, max_reviews = book_review_counts.most_common(1)[0]
        most_reviewed_title = book_id_to_title.get(most_reviewed_book_id, "Unknown Title")
        print(f"\nMost reviewed book: {most_reviewed_title}")
        print(f"Book ID: {most_reviewed_book_id}")
        print(f"Number of reviews: {max_reviews:,}")
    
    # Duplicate title analysis
    print("\n" + "=" * 80)
    print("DUPLICATE TITLE ANALYSIS")
    print("=" * 80)
    
    # Group book IDs by title
    title_to_book_ids = defaultdict(list)
    for book in books:
        title_to_book_ids[book['title']].append(book['book_id'])
    
    # Find duplicates
    duplicate_titles = {title: book_ids for title, book_ids in title_to_book_ids.items() 
                       if len(book_ids) > 1}
    
    print(f"Total unique titles: {len(title_to_book_ids):,}")
    print(f"Titles with duplicate book IDs: {len(duplicate_titles):,}")
    
    # Count total books involved in duplicates
    total_duplicate_books = sum(len(book_ids) for book_ids in duplicate_titles.values())
    print(f"Total books with duplicate titles: {total_duplicate_books:,}")
    
    # Show some examples of duplicates
    if duplicate_titles:
        print("\nTop 10 titles with most duplicate book IDs:")
        sorted_duplicates = sorted(duplicate_titles.items(), 
                                  key=lambda x: len(x[1]), 
                                  reverse=True)
        
        for i, (title, book_ids) in enumerate(sorted_duplicates[:10], 1):
            print(f"\n{i:2d}. {title[:70]}")
            print(f"    Number of book IDs: {len(book_ids)}")
            print(f"    Book IDs: {', '.join(book_ids[:5])}", end="")
            if len(book_ids) > 5:
                print(f" ... (+{len(book_ids)-5} more)")
            else:
                print()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total books in dataset: {len(books):,}")
    print(f"Total unique book titles: {len(title_to_book_ids):,}")
    print(f"Total reviews: {len(reviews):,}")
    print(f"Total unique users: {len(user_review_counts):,}")
    print(f"Average reviews per book (with reviews): {len(reviews) / books_with_reviews:.2f}")
    print(f"Average reviews per user: {len(reviews) / len(user_review_counts):.2f}")
    print("=" * 80)
    
    # DEDUPLICATION
    print("\n" + "=" * 80)
    print("CREATING DEDUPLICATED DATASETS")
    print("=" * 80)
    
    # For each title with duplicates, find the book_id with the most reviews
    # Create mapping: old_book_id -> canonical_book_id
    book_id_mapping = {}
    canonical_book_ids = set()
    
    for title, book_ids in title_to_book_ids.items():
        if len(book_ids) == 1:
            # No duplicates, use the only book_id
            canonical_book_ids.add(book_ids[0])
        else:
            # Find which book_id has the most reviews
            book_id_review_counts = [(book_id, book_review_counts.get(book_id, 0)) 
                                    for book_id in book_ids]
            # Sort by review count (descending), then by book_id for consistency
            book_id_review_counts.sort(key=lambda x: (-x[1], x[0]))
            canonical_id = book_id_review_counts[0][0]
            canonical_book_ids.add(canonical_id)
            
            # Map all book_ids to the canonical one
            for book_id in book_ids:
                book_id_mapping[book_id] = canonical_id
            
            # print(f"\nTitle: {title[:70]}")
            # print(f"  Canonical book_id: {canonical_id} ({book_id_review_counts[0][1]} reviews)")
            # if len(book_ids) > 1:
            #     print(f"  Merged book_ids: {', '.join([bid for bid in book_ids if bid != canonical_id])}")
    
    print(f"\nTotal canonical book IDs: {len(canonical_book_ids):,}")
    print(f"Total book IDs that will be merged: {len(book_id_mapping) - len(canonical_book_ids):,}")
    
    # Create deduplicated books dataset
    deduplicated_books = []
    for book in books:
        book_id = book['book_id']
        canonical_id = book_id_mapping.get(book_id, book_id)
        
        # Only keep the book if it's the canonical version
        if book_id == canonical_id:
            deduplicated_books.append(book)
    
    print(f"\nOriginal books: {len(books):,}")
    print(f"Deduplicated books: {len(deduplicated_books):,}")
    print(f"Books removed: {len(books) - len(deduplicated_books):,}")
    
    # Create deduplicated reviews dataset with updated book_ids
    deduplicated_reviews = []
    updated_count = 0
    
    for review in reviews:
        old_book_id = review['book_id']
        new_book_id = book_id_mapping.get(old_book_id, old_book_id)
        
        if old_book_id != new_book_id:
            updated_count += 1
        
        # Create updated review with new book_id
        updated_review = review.copy()
        updated_review['book_id'] = new_book_id
        deduplicated_reviews.append(updated_review)
    
    print(f"\nTotal reviews: {len(reviews):,}")
    print(f"Reviews with updated book_id: {updated_count:,}")
    
    # Save deduplicated datasets
    print("\n" + "=" * 80)
    print("SAVING DEDUPLICATED DATASETS")
    print("=" * 80)
    save_json_lines(deduplicated_books, 'goodreads_books_children_dedup.json')
    save_json_lines(deduplicated_reviews, 'goodreads_reviews_children_dedup.json')
    
    # Verify deduplication
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Check for duplicate titles in deduplicated books
    dedup_title_to_ids = defaultdict(list)
    for book in deduplicated_books:
        dedup_title_to_ids[book['title']].append(book['book_id'])
    
    duplicates_remaining = sum(1 for ids in dedup_title_to_ids.values() if len(ids) > 1)
    print(f"Duplicate titles remaining in deduplicated books: {duplicates_remaining}")
    print(f"Unique titles in deduplicated books: {len(dedup_title_to_ids):,}")
    
    # Verify all review book_ids exist in deduplicated books
    dedup_book_ids = {book['book_id'] for book in deduplicated_books}
    review_book_ids = {review['book_id'] for review in deduplicated_reviews}
    missing_book_ids = review_book_ids - dedup_book_ids
    
    print(f"Book IDs in deduplicated reviews: {len(review_book_ids):,}")
    print(f"Book IDs in deduplicated books: {len(dedup_book_ids):,}")
    print(f"Review book IDs missing from books dataset: {len(missing_book_ids):,}")
    
    print("\n" + "=" * 80)
    print("DEDUPLICATION COMPLETE!")
    print("=" * 80)
    print("Output files:")
    print("  - goodreads_books_children_dedup.json")
    print("  - goodreads_reviews_children_dedup.json")
    print("=" * 80)

if __name__ == "__main__":
    main()