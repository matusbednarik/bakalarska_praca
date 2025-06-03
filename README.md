Bakalárska práca na tému WEB CRAWLER PRE ZBER INFORMÁCIÍ V BEZPEČNOSTNEJ DOMÉNE.
Repozitár obsahuje päť hlavných adresárov.
1. focused_crawler - obsahuje kód na využitie scrapy web crawlera. Hlavná logika je v súbore cybersecurity_articles.py a výstup je crawledWebsites.json.
                   - crawledWebsitesStrba.json obsahuje články z nadväzujúcej práce na rovnakú tému z FEI STU.
2. BERT Classifier - adresár preprocessing obsahuje kód na očistenie a spracovanie dát, ktoré budú ako príloha pri odovzdaní (nezmestia sa na GitHub)
                   - adresár scripts obsahuje skript train_BERT.py, ktorý pri spustení sa natrénuje na článkoch z datasetu https://github.com/cypher-07/Cybersecurity-News-Article-Dataset
                   - adresár label_articles v adresári scripts obsahuje program na klasifikovanie článkov z web crawlera pomocou novo natrénovaného BERT klasifikátora. Výstup je bert_labeled_articles.csv (naše aktuálne získané články) a bert_labeled_articlesStrba.csv obsahuje články z nadväzujúcej práce.
                   - adresár filter_articles_by_label obsahuje jednoduchý skript na odfiltrovanie všetkých článkov, ktoré nepatria do kategórie Malware. Výstup na našich článkoch je filtered_malware_articles.csv a filtered_malware_articlesStrba.csv je výstup na článkoch z nadväzujúcej práce.
3. spacy_with_ontology - obsahuje extract_malware_entities.py, ktorý ako vstup dostane crawledWebsites.json a ako výstup dá extrahované entity do malware_entities.csv
                       - MALont.owl obsahuje malvérovú ontológiu
4. malware_name_extraction - obsahuje kód na finetuning spaCy modelu datasetom [CyNER](https://huggingface.co/datasets/PranavaKailash/CyNER2.0_augmented_dataset)
                           - convert_cyner_to_spacy.py konvertuje spomínaný dataset na spacy formát. Výstup je malware_test.spacy, malware_train.spacy a malware_validation.spacy 
                           - train_model_cyner_data.py + config.cfg slúzia na trénovanie modelu
                           - find_malware.py aplikuje nový natrénovaný model na crawledWebsites.json a dá výsledok do výstupu articles_with_malware_entities_spacy.csv
5. CyNER-2.0-DeBERTa-v3-base - obsahuje skript find_cybersecurity_names.py, ktorý využije natrénovaný model https://huggingface.co/PranavaKailash/CyNER-2.0-DeBERTa-v3-base aby našiel entity ohľadom kyberbezpečnosti a dá ich do výstupu articles_with_all_entities.py, kde každý článok obsahuje zoznam nájdených entít.

6. Autor: Matúš Bednařík
