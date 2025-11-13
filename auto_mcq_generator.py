import warnings
warnings.filterwarnings("ignore")
text = """Python was created by Guido van Rossum and first released in 1991. 
It is widely used at companies like Google, NASA, and Quora for artificial intelligence and web development."""

from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")


input_text = "generate question: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_beams=2, early_stopping=True)

question = tokenizer.decode(outputs[0])
print("Generated Question:", question)

print("Script finished.")

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
answers = [ent.text for ent in doc.ents]
print("Named Entity Answers:", answers)

from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet

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

if answers:
    correct_answer = answers[0]
    distractors = generate_distractors(correct_answer)
    print(f"Correct Answer: {correct_answer}")
    print(f"Distractors: {distractors}")
else:
    print("No answers found for distractor generation.")
