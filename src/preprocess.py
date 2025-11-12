import re
def preprocess_text(text):

    text = text.lower()  # Convert text to lowercase

    # Remove HTML tags

    text = re.sub(r"<.*?>", "", text)

    # Remove links from text

    text = re.sub(r"http\S+|www\S+", "", text)

    return text.strip()  # Strip remaining whitespace around text

