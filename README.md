# WEB CRAWLER PRE ZBER INFORMÁCIÍ V BEZPEČNOSTNEJ DOMÉNE  
**Autor: Matúš Bednařík**

Tento repozitár implementuje riešenie rozdelené do piatich hlavných adresárov.

---

## Obsah

1. [focused_crawler](#1-focused_crawler)  
2. [BERT Classifier](#2-bert-classifier)  
3. [spacy_with_ontology](#3-spacy_with_ontology)  
4. [malware_name_extraction](#4-malware_name_extraction)  
5. [CyNER-2.0-DeBERTa-v3-base](#5-cyner-20-deberta-v3-base)  

---

## 1. `focused_crawler`

Adresár obsahuje kód pre web crawler implementovaný pomocou knižnice **Scrapy**.

- **Hlavná logika**  
  - Súbor: `cybersecurity_articles.py`  
  - Popis: Zabezpečuje prehľadávanie vybraných webových stránok a ukladanie výsledkov.  
  - Výstup: `crawledWebsites.json`

- **Doplnkový súbor**  
  - `crawledWebsitesStrba.json` – obsahuje články zo súvisiacej bakalárskej práce realizovanej na FEI STU.

---

## 2. `BERT Classifier`

Tento adresár je venovaný klasifikácii článkov pomocou modelu BERT.

### 2.1 `preprocessing`

Obsahuje skripty na čistenie a prípravu dát.  
> Dáta z tohto kroku sú priložené len pri odovzdávaní práce a nezmestia sa na GitHub.

### 2.2 `scripts/train_BERT.py`

- **Popis**: Tréning BERT klasifikátora na článkoch z datasetu  
- **Dataset**: [Cybersecurity-News-Article-Dataset](https://github.com/cypher-07/Cybersecurity-News-Article-Dataset)

### 2.3 `scripts/label_articles`

Skript na klasifikáciu článkov získaných web crawlerom pomocou novo natrénovaného BERT modelu.

- **Výstupy**:  
  - `bert_labeled_articles.csv` – klasifikované aktuálne články  
  - `bert_labeled_articlesStrba.csv` – klasifikované články z nadväzujúcej práce

### 2.4 `filter_articles_by_label`

Jednoduchý skript na odfiltrovanie článkov podľa kategórie **Malware**.

- **Výstupy**:  
  - `filtered_malware_articles.csv` – filtrované články z aktuálneho datasetu  
  - `filtered_malware_articlesStrba.csv` – filtrované články z nadväzujúcej práce

---

## 3. `spacy_with_ontology`

Zameraný na extrakciu entít z článkov pomocou ontológie.

- **`extract_malware_entities.py`**  
  - **Vstup**: `crawledWebsites.json`  
  - **Výstup**: `malware_entities.csv`  

- **`MALont.owl`**  
  - Ontológia pre oblasť malvéru

---

## 4. `malware_name_extraction`

Tréning a aplikácia **spaCy** modelu na detekciu názvov malvéru.

- **`convert_cyner_to_spacy.py`**  
  - Konverzia datasetu [CyNER](https://huggingface.co/datasets/PranavaKailash/CyNER2.0_augmented_dataset) do formátu spaCy  
  - Výstupy:  
    - `malware_train.spacy`  
    - `malware_validation.spacy`  
    - `malware_test.spacy`

- **`train_model_cyner_data.py`** + `config.cfg`  
  - Tréning modelu na spracovanom CyNER dátovom sete

- **`find_malware.py`**  
  - Aplikácia novo natrénovaného modelu na dáta z `crawledWebsites.json`  
  - Výstup: `articles_with_malware_entities_spacy.csv`

---

## 5. `CyNER-2.0-DeBERTa-v3-base`

Extrakcia entít pomocou predtrénovaného modelu **CyNER-2.0-DeBERTa-v3-base**.

- **`find_cybersecurity_names.py`**  
  - Využíva model [CyNER-2.0-DeBERTa-v3-base](https://huggingface.co/PranavaKailash/CyNER-2.0-DeBERTa-v3-base)  
  - Hľadá entity z oblasti kyberbezpečnosti  

- **Výstup**: `articles_with_all_entities.py`  
  - Každý článok obsahuje zoznam extrahovaných entít

---

