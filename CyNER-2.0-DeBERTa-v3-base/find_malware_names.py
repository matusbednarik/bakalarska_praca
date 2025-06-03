import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm

# 1) Load your pre-filtered articles
input_path  = r'C:\Users\HP\Documents\bakalarka\BERT Classifier\scripts\filter_articles_by_label\filtered_malware_articles.csv'
df = pd.read_csv(input_path)

# 2) Download & initialize the CyNER 2.0 model
#    (requires: pip install transformers torch)
MODEL_ID = "PranavaKailash/CyNER-2.0-DeBERTa-v3-base"  # ↳ https://huggingface.co/PranavaKailash/CyNER-2.0-DeBERTa-v3-base
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

# 3) Build an NER pipeline that groups sub-tokens into full entities
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # collapse B- and I- tokens into one span
    device=-1                      
)

# 4) Define a helper to extract all entity types
def extract_all_entities(text, score_thresh=0.7):
    ents = ner(text)
    entities = {}
    
    for ent in ents:
        # Get the entity label
        label = ent.get("entity_group", ent.get("entity", ""))
        
        # Extract the entity type (removing B- or I- prefix)
        if "-" in label:
            entity_type = label.split("-")[-1]
        else:
            entity_type = label
            
        # Skip entities below the confidence threshold
        if ent["score"] < score_thresh:
            continue
            
        # Add to appropriate category
        if entity_type not in entities:
            entities[entity_type] = []
            
        # Only store the text without the score
        entities[entity_type].append(ent["word"])
    
    return entities

# 5) Apply to each row
tqdm.pandas()
df["entities"] = df["Preprocessed_Content"].progress_apply(lambda txt: extract_all_entities(txt, score_thresh=0.7))

# 6) Save results
output_path = r'C:\Users\HP\Documents\bakalarka\CyNER-2.0-DeBERTa-v3-base\articles_with_all_entities.csv'
df.to_csv(output_path, index=False)

print(f"✅ Done! Wrote {len(df)} rows with all extracted entities to:\n  {output_path}")
