# Use this script if your data is already normalised by UMLS

import csv
import re
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher
from negspacy.negation import Negex
from negspacy.termsets import termset
from tqdm import tqdm
import pandas as pd

# Define the terms related to conditions and their corresponding normalization strings
normalization_terms = {
    "Ulcerative Colitis": "ulcerative_colitis",
    "Crohn's Disease": "crohns_disease",
    "IBD": "inflammatory_bowel_disease",
    "Inflammatory Bowel Disease": "inflammatory_bowel_disease",
    "Proctitis": "proctitis",
    "Collagenous Colitis": "collagenous_colitis",
    "Microscopic Colitis": "microscopic_colitis",
    "Lymphocytic Colitis": "lymphocytic_colitis"
}

# Compile regex patterns for normalization terms
compiled_patterns = {term: re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE) for term in normalization_terms}

# Function to clean and normalize documents using compiled regex patterns
def clean_and_normalize_documents(documents, compiled_patterns, normalization_terms):
    processed_documents = []
    for doc in documents:
        if not isinstance(doc, str):
            doc = str(doc)
        for term, pattern in compiled_patterns.items():
            normalized = normalization_terms[term]
            doc = pattern.sub(normalized, doc)
        processed_documents.append(doc)
    return processed_documents

# Define token-based patterns for the normalized terms
token_patterns = {term.upper(): [{"LOWER": term}] for term in normalization_terms.values()}

# Extended custom negation terms to strengthen the negation detection
custom_pseudo_negations = ["without a trace of", "no evidence of"]
custom_preceding_negations = [
    "no sign of", "not suggestive of", "not consistent with", "without evidence of",
    "free from", "absent", "negative for", "none", "rule out", "does not show",
    "unlikely", "not", "never", "neither", "cannot be seen", "not identified", "no signs of",
    "no indication of", "no findings of", "no proof of", "no traces of"
]
custom_following_negations = []
custom_terminations = ["but", "however", "although", "despite"]

# Get the standard negation terms from Negex
ts = termset("en").get_patterns()

# Combine standard and custom negation terms
negation_terms = {
    "pseudo_negations": ts['pseudo_negations'] + custom_pseudo_negations,
    "preceding_negations": ts['preceding_negations'] + custom_preceding_negations,
    "following_negations": ts['following_negations'] + custom_following_negations,
    "termination": ts['termination'] + custom_terminations
}

# Initialize and configure the spaCy model
def initialize_nlp():
    nlp = spacy.blank("en")
    
    # Add custom preprocessing component
    @spacy.Language.component("custom_preprocess")
    def preprocess(doc):
        return nlp.make_doc(doc.text.lower())
    nlp.add_pipe("custom_preprocess", first=True)

    # Add sentencizer for sentence boundaries
    nlp.add_pipe("sentencizer")

    # Set up the Matcher
    matcher = Matcher(nlp.vocab)
    for label, pattern in token_patterns.items():
        matcher.add(label, [pattern])

    # Add the matcher to the pipeline
    @spacy.Language.component("custom_matcher")
    def custom_matcher(doc):
        matches = matcher(doc)
        spans = [Span(doc, start, end, label=label) for match_id, start, end in matches]
        doc.ents = spans
        return doc
    nlp.add_pipe("custom_matcher", after="sentencizer")

    # Set up Negex with custom negation terms
    negex = Negex(
        nlp,
        name="negex",
        neg_termset=negation_terms,
        ent_types=list(token_patterns.keys()),
        extension_name="negex",
        chunk_prefix={"no", "without"}
    )
    nlp.add_pipe("negex", last=True)

    return nlp

# Function to flag true mentions of conditions
def flag_true_mentions(doc):
    true_mentions = []
    for ent in doc.ents:
        if not ent._.negex:
            true_mentions.append(ent.text)
    return ', '.join(true_mentions) if true_mentions else 'None'

# Function to process documents and handle negation detection
def process_documents(nlp, documents):
    results = []
    processed_indices = []  # Keep track of indices of the documents processed
    for index, doc_text in enumerate(tqdm(documents, desc="Processing documents")):
        if pd.isna(doc_text):  # Check if the document is NaN
            continue  # Skip this document
        doc = nlp(str(doc_text))  # Ensure it's a string
        condition_flag = False
        doc_info = []
        for ent in doc.ents:
            if not ent._.negex:
                doc_info.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "is_negated": ent._.negex
                })
                condition_flag = True
        results.append({"document_condition_flag": condition_flag, "entities": doc_info})
        processed_indices.append(index)  # Store the index of the processed document
    return results, processed_indices

# Initialize the NLP model
nlp = initialize_nlp()

# Load the clinic letters dataframe
clinic_letters_df = pd.read_csv('path_to_clinic_letters.csv')

# Check the column names in the clinic letters dataframe
print(clinic_letters_df.columns)

# Extract the combined comments
documents = clinic_letters_df['clean_content'].tolist()

# Clean and normalize the documents
cleaned_documents = clean_and_normalize_documents(documents, compiled_patterns, normalization_terms)

# Process the documents with the NLP pipeline
condition_info, processed_indices = process_documents(nlp, cleaned_documents)

# Extract flags and update DataFrame
flags = [info['document_condition_flag'] for info in condition_info]
clinic_letters_df.loc[clinic_letters_df.index[processed_indices], 'Condition_Suggestive'] = flags

# Save the results to a CSV file
clinic_letters_df.to_csv('processed_clinic_letters_with_flags.csv', index=False)

# Print the first few rows to verify the results
print(clinic_letters_df.head())

# Print the value counts of the 'Condition_Suggestive' column
print(clinic_letters_df['Condition_Suggestive'].value_counts())
