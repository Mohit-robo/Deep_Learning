# Named Entity Recognition (NER) 🏷️

Extract and classify named entities (people, organizations, locations) from text.

## 📚 Overview

NER is a sequence labeling task where each token is assigned a label indicating entity type or non-entity status.

**Example**:
```
Input: "Apple was founded by Steve Jobs in California"
Output:
- Apple: Organization
- Steve Jobs: Person
- California: Location
```

---

## 📂 Files

- **ner.ipynb**: Standard NER using existing libraries
- **ner_custom.ipynb**: Training custom NER on annotated data

---

## 🔧 NER Process

```
Text → Tokenize → Feature Extraction → Classification → Output
```

**Key Steps**:
1. Tokenize text into words/subwords
2. Extract features (embeddings, POS tags, etc.)
3. Classify each token as entity or O (other)
4. Combine consecutive tokens into entities

---

## 🎯 Entity Tags

Common BIO (Begin-Inside-Other) format:
- **O**: Non-entity
- **B-PER**: Beginning of Person entity
- **I-PER**: Inside Person entity
- **B-ORG**: Beginning of Organization
- **I-ORG**: Inside Organization
- **B-LOC**: Beginning of Location
- **I-LOC**: Inside Location

**Example**:
```
Apple   → B-ORG
was     → O
founded → O
by      → O
Steve   → B-PER
Jobs    → I-PER
in      → O
California → B-LOC
```

---

## 🚀 Quick Start

```python
# Using standard library (ner.ipynb)
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple was founded by Steve Jobs")

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Custom training (ner_custom.ipynb)
# See notebook for full implementation
```

---

## 🛠️ Use Cases

- **Resume parsing**: Extract skills, experiences
- **News processing**: Identify people, organizations, locations
- **Question answering**: Extract entities for QA systems
- **Knowledge graphs**: Build entity relationships

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: NLP basics, PyTorch or spaCy