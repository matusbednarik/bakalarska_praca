import json

def count_unique_links(json_path: str) -> int:
    """Load the JSON at json_path and return the number of unique 'link' fields."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Collect all non-empty link values into a set
    unique_links = {item.get('link') for item in data if item.get('link')}
    return len(unique_links)

if __name__ == '__main__':
    path = 'C:\\Users\HP\\Documents\\bakalarka\\cybersecurity_terms\\crawledWebsites.json'
    total = count_unique_links(path)
    print(f'Total unique articles (links) in {path}: {total}')