import urllib.request
import json
import re
import os

titles = [
    "Artificial_intelligence", "Machine_learning", "Deep_learning",
    "Natural_language_processing", "Python_(programming_language)",
    "Data_science", "Reinforcement_learning", "Computer_vision"
]

def fetch_wiki(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={title}&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        pages = data['query']['pages']
        for page_id in pages:
            return pages[page_id].get('extract', '')
    return ""

def main():
    text = ""
    for t in titles:
        print(f"Fetching {t}...")
        try:
            text += fetch_wiki(t) + "\n\n"
        except Exception as e:
            print(f"Error fetching {t}: {e}")

    # Clean up wiki formatting somewhat
    text = re.sub(r'==+ .*? ==+', '', text)
    text = re.sub(r'\[.*?\]', '', text) # remove citations [1]
    
    # Add a conversational starter so the model knows basic greetings
    conversational_starter = """
    Hello there, how are you mr josh. I am doing very well today.
    The goal of this project is to build an artificial intelligence using machine learning.
    I love python programming language and deep learning neural networks.
    """
    
    final_text = conversational_starter + text

    os.makedirs("data", exist_ok=True)
    out_path = "data/tech_corpus.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"Saved {len(final_text.split())} words to {out_path}")

if __name__ == "__main__":
    main()
