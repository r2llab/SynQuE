import re

def clean_text_output(text_output: str) -> str:
    """Clean the text output"""
    # First extract content between triple backticks if present
    if "```" in text_output:
        parts = text_output.split("```")
        # Get the content between first and second ```
        if len(parts) >= 3:
            text_output = parts[1].strip()
    else:
        text_output = text_output.strip()
        if text_output.lower().startswith('json'):
            text_output = text_output[len("json"):].strip()
        return fix_double_quotes(text_output)
    
    # Remove language identifier if it starts with 'j'
    if text_output.lower().startswith('json'):
        text_output = text_output[len("json"):].strip()
    
    return fix_double_quotes(text_output)

def fix_double_quotes(text_output: str) -> str:
    """Fix double quotes in the text output"""
    # first capture strings within each ""
    pattern = r'"(.*?)"\,'
    matches = re.findall(pattern, text_output)
    for match in matches:
        # Replace double quotes with a single escaped double quote
        # escaped_match = match.replace('"', r'\"')
        escaped_match = match
        # Replace the original match with the escaped version in the text output
        text_output = text_output.replace(f'"{match}"', f'"{escaped_match}"')

    return text_output