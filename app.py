import streamlit as st
import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet

# Load models once
nlp = spacy.load("en_core_web_sm")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)

def generate_distractors(correct_answer, top_k=3):
    synonyms = get_synonyms(correct_answer)
    if not synonyms:
        return ["OptionA", "OptionB", "OptionC"]
    candidates = synonyms
    embeddings = embedder.encode(candidates + [correct_answer], convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings[-1], embeddings[:-1])[0]
    sorted_pairs = sorted(zip(candidates, cos_scores.cpu().tolist()), key=lambda x: x[1])
    distractors = [pair[0] for pair in sorted_pairs[:top_k]]
    return distractors

st.title("Automatic MCQ Generator")

text = st.text_area("Enter text to generate MCQs:", height=200)

if st.button("Generate MCQ"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Generate question
        input_text = "generate question: " + text
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_beams=2, early_stopping=True)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answers
        doc = nlp(text)
        answers = [ent.text for ent in doc.ents]

        if answers:
            correct_answer = answers[0]
            distractors = generate_distractors(correct_answer)
        else:
            correct_answer = "N/A"
            distractors = ["OptionA", "OptionB", "OptionC"]

        # Display MCQ
        st.subheader("Generated Question:")
        st.write(question)
        st.subheader("Choices:")
        st.write(f"A. {correct_answer}  (Correct Answer)")
        for i, option in enumerate(distractors, start=2):
            st.write(f"{chr(64+i)}. {option}")
