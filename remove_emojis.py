import json
import re

def remove_emojis(text):
    # Remove all emojis using a more comprehensive pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Remove emojis from markdown cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            if 'source' in cell and isinstance(cell['source'], list):
                cell['source'] = [remove_emojis(line) for line in cell['source']]
    
    # Write the cleaned notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    notebook_path = "ML_Data_Preparation_Example.ipynb"
    clean_notebook(notebook_path)
    print(f"Emojis removed from {notebook_path}")
