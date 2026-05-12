"""Generates a synthetic multilingual product catalog (PT/EN).

Produces items representative of Brazilian e-commerce (Mercado Livre-style),
intentionally mixing Portuguese and English content to exercise cross-lingual retrieval.
"""

import json
import random

CATEGORIES = ["eletrônicos", "moda", "casa", "esportes", "beleza"]

PT_TEMPLATES = [
    ("{cat} - {adj} {noun}", "Produto de alta qualidade na categoria {cat}. {desc}"),
    ("Melhor {noun} para {use}", "{adj} e durável. Ideal para {use}. Entrega rápida."),
    ("{noun} {adj} importado", "Importado com garantia. Categoria: {cat}. {desc}"),
]

EN_TEMPLATES = [
    ("{adj} {noun} for {use}", "High-quality {cat} product. {desc}"),
    ("Premium {noun} - {adj}", "Imported with warranty. Category: {cat}. {desc}"),
]

ADJECTIVES_PT = ["premium", "profissional", "compacto", "ergonômico", "resistente"]
NOUNS_PT = ["fone de ouvido", "tênis", "mochila", "câmera", "relógio"]
USES_PT = ["academia", "trabalho", "viagem", "casa", "aventura"]
DESCS_PT = [
    "Desenvolvido com tecnologia avançada.",
    "Material de primeira linha.",
    "Aprovado por especialistas.",
]

ADJECTIVES_EN = ["premium", "professional", "compact", "ergonomic", "durable"]
NOUNS_EN = ["headphones", "sneakers", "backpack", "camera", "smartwatch"]
USES_EN = ["gym", "work", "travel", "home", "outdoor"]


def generate_catalog(n: int = 5000) -> list[dict]:
    items = []
    for i in range(n):
        lang = "pt" if random.random() < 0.7 else "en"
        cat = random.choice(CATEGORIES)

        if lang == "pt":
            tmpl_title, tmpl_desc = random.choice(PT_TEMPLATES)
            adj = random.choice(ADJECTIVES_PT)
            noun = random.choice(NOUNS_PT)
            use = random.choice(USES_PT)
            desc = random.choice(DESCS_PT)
            title = tmpl_title.format(cat=cat, adj=adj, noun=noun, use=use)
            description = tmpl_desc.format(cat=cat, adj=adj, noun=noun, use=use, desc=desc)
        else:
            tmpl_title, tmpl_desc = random.choice(EN_TEMPLATES)
            adj = random.choice(ADJECTIVES_EN)
            noun = random.choice(NOUNS_EN)
            use = random.choice(USES_EN)
            desc = "Made with premium materials."
            title = tmpl_title.format(adj=adj, noun=noun, use=use)
            description = tmpl_desc.format(cat=cat, adj=adj, noun=noun, desc=desc)

        items.append({
            "id": f"item_{i:05d}",
            "title": title,
            "description": description,
            "category": cat,
            "language": lang,
        })

    return items


if __name__ == "__main__":
    catalog = generate_catalog(5000)
    with open("data/catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(catalog)} items")
    pt_count = sum(1 for i in catalog if i["language"] == "pt")
    print(f"  PT: {pt_count} | EN: {len(catalog) - pt_count}")
