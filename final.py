# %%
import tkinter as tk
from tkinter import filedialog, messagebox
import spacy
from transformers import MarianMTModel, MarianTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy import displacy
import nltk
from summarizer import Summarizer


# %%
summarizer = Summarizer()

# %%
nlp_en = spacy.load("en_core_web_sm")
nlp_ar = spacy.load("xx_ent_wiki_sm")

# %%
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# %%

# Function to perform sentiment analysis
def perform_sentiment_analysis(document_text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(document_text)
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# %%
# Function to perform named entity recognition
def perform_ner(document_text):
    doc = nlp_en(document_text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# %%
def perform_text_summarization(document_text, language):
    if language == 'en':
        summary = summarizer(document_text)
    else:
        translated_text = translate_to_arabic(document_text)
        summary = summarizer(translated_text)
    return summary

# %%
def translate_to_arabic(english_text):
    input_ids = tokenizer.encode(english_text, return_tensors="pt")
    translated_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text



# %%
# Function to open and process the document
def open_document():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            document_text = file.read()
        # Perform NLP tasks
        sentiment = perform_sentiment_analysis(document_text)
        entities = perform_ner(document_text)
        english_summary = perform_text_summarization(document_text, 'en')
        arabic_summary = perform_text_summarization(document_text, 'ar')

        # Display results
        messagebox.showinfo("Sentiment", f"Sentiment: {sentiment}")
        messagebox.showinfo("Named Entities", f"Named Entities: {entities}")
        messagebox.showinfo("English Summary", f"English Summary:\n{english_summary}")
        messagebox.showinfo("Arabic Summary", f"Arabic Summary:\n{arabic_summary}")


# %%
# Create GUI
nltk.download('vader_lexicon')

root = tk.Tk()
root.title("Document Analysis App")

# Create and pack widgets
open_button = tk.Button(root, text="Open Document", command=open_document)
open_button.pack(pady=20)

root.mainloop()


