import os
import csv
import requests
from collections import deque

# If you don't need sentence splitting, you don't need NLTK here.
# But I'll leave the import for reference if you want to do sentences too:
# import nltk
# from nltk.tokenize import sent_tokenize

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def get_category_members(category_title, cmcontinue=None, limit="max"):
    """
    Retrieve a batch of category members (pages + subcategories)
    from a single category using the MediaWiki API.
    """
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_title,
        "format": "json",
        "cmlimit": limit,
    }
    if cmcontinue:
        params["cmcontinue"] = cmcontinue

    resp = requests.get(WIKIPEDIA_API_URL, params=params)
    data = resp.json()

    if "error" in data:
        print(f"Error from API: {data['error']}")
        return [], None

    categorymembers = data.get("query", {}).get("categorymembers", [])
    next_cmcontinue = data.get("continue", {}).get("cmcontinue")

    return categorymembers, next_cmcontinue

def collect_all_pages_from_category(root_categories, max_pages=100000):
    """
    Collect up to 'max_pages' Wikipedia pages (ns=0) 
    starting from 'root_categories' and traversing subcategories (ns=14).
    """
    categories_to_explore = deque(root_categories)
    visited_categories = set()
    page_titles = set()
    
    while categories_to_explore and len(page_titles) < max_pages:
        cat = categories_to_explore.popleft()
        if cat in visited_categories:
            continue
        visited_categories.add(cat)

        cmcontinue = None
        while True:
            members, cmcontinue = get_category_members(cat, cmcontinue=cmcontinue, limit="max")
            if not members:
                break

            for m in members:
                ns = m["ns"]
                title = m["title"]
                
                # Subcategory => queue for further exploration
                if ns == 14:  # 14 = Category namespace
                    if title not in visited_categories:
                        categories_to_explore.append(title)

                # Page => add to our list of article titles
                elif ns == 0:  # 0 = Main/Article namespace
                    page_titles.add(title)
                    if len(page_titles) >= max_pages:
                        break
            
            if len(page_titles) >= max_pages or not cmcontinue:
                break
    
    return page_titles

def get_page_text(page_title):
    """
    Fetch the plain-text extract for a given Wikipedia page title.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": page_title,
        "format": "json",
    }
    resp = requests.get(WIKIPEDIA_API_URL, params=params)
    data = resp.json()

    pages_data = data.get("query", {}).get("pages", {})
    for page_id, page_info in pages_data.items():
        if page_id == "-1":
            return ""  # page not found
        return page_info.get("extract", "")
    
    return ""

def build_two_paragraph_pair(text, min_word_length=5):
    """
    Split the text by paragraphs and ONLY return a single pair (p1, p2)
    from the first two paragraphs, if available and not too short.
    
    Returns None if we can't build a valid pair (i.e., fewer than 2 paragraphs 
    or paragraphs too short).
    """
    # Split by double newlines as a simple paragraph heuristic
    paragraphs = text.split("\n\n")
    # Clean up whitespace and skip empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # We need at least 2 paragraphs
    if len(paragraphs) < 2:
        return None
    
    p1, p2 = paragraphs[0], paragraphs[1]
    
    # Check minimal word count
    if len(p1.split()) < min_word_length or len(p2.split()) < min_word_length:
        return None
    
    return (p1, p2)

def create_wiki_cl(
    root_categories = ["Category:Human diseases and disorders","Category:Clinical medicine","Category:Disability","Category:Infectious diseases", "Category:Pathology"],
    max_pages = 200000,
    output_dir = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs/"
):
    """
    1) Collect up to 'max_pages' Wikipedia pages from root categories (including subcats).
    2) For each page, get the first two paragraphs.
    3) If valid, store them as (paragraph1, paragraph2) in a CSV.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "wiki_contrastive_dataset.csv")
    
    # 1) Collect article page titles
    print(f"Collecting up to {max_pages} pages from {root_categories} ...")
    page_titles = collect_all_pages_from_category(root_categories, max_pages=max_pages)
    print(f"Collected {len(page_titles)} page titles.")
    
    # 2) Build dataset (list of (p1, p2))
    all_pairs = []
    for idx, title in enumerate(page_titles):
        text = get_page_text(title)
        if not text:
            continue
        
        pair = build_two_paragraph_pair(text, min_word_length=5)
        if pair is None:
            # Skip this title if we cannot form a valid two-paragraph pair
            continue
        
        all_pairs.append(pair)

        # Optional: progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(page_titles)} pages... "
                  f"Current valid pair count: {len(all_pairs)}")
    
        # 3) Save to CSV
        if (idx + 1) % 1000 == 0:
            print(f"Saving {len(all_pairs)} two-paragraph pairs to CSV at: {output_csv_path}")
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sentence1", "sentence2"])  # header
                for p1, p2 in all_pairs:
                    writer.writerow([p1, p2])
    print(f"Saving {len(all_pairs)} two-paragraph pairs to CSV at: {output_csv_path}")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence1", "sentence2"])  # header
        for p1, p2 in all_pairs:
            writer.writerow([p1, p2])
    print("Done!")
    return output_csv_path


if __name__ == "__main__":
    # Example usage:
    # This will collect up to 5,000 articles from the "Medicine" category & subcategories,
    # build consecutive-sentence pairs, and save them to 'contrastive_dataset.csv' in current dir.
    
    root_cats = ["Category:Human diseases and disorders","Category:Clinical medicine","Category:Disability","Category:Infectious diseases", "Category:Pathology"]
    max_pages = 200000
    output_directory = "/home/skyfury/projects/def-mahyarh/skyfury/CTMEDBERT/CTMEDBERT/data/csvs/"
    
    csv_path = create_wiki_cl()
    print(f"CSV saved to: {csv_path}")
