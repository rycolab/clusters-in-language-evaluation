import time
from datasets import load_dataset
from collections import Counter
from goose3 import Goose
from lxml.etree import ParserError


author_dataset = load_dataset('Fraser/news-category-dataset', split='train')
author_labels = [x.split(",")[0] for x in author_dataset['authors']]
counts = sorted(Counter(author_labels).items(), key=lambda item: item[1], reverse=True)

authors = set([a for a, n in counts if n >=400])
author_dataset = load_dataset('Fraser/news-category-dataset', split='test')
author_labels = [x.split(",")[0] for x in author_dataset['authors']]
author_dataset_filtered = [(author, article) for author, article in zip(author_labels, author_dataset['link']) if (author and author in authors)]

author_dataset_filtered = author_dataset_filtered[:5000]
articles = []
g = Goose()
for _, l in author_dataset_filtered:
    # First error might simply come from issuing too many requests close to each other
    try:
        articles.append(g.extract(url=l).cleaned_text.replace('\n',' '))
    except ParserError as e:
        time.sleep(65)
        # Second error means there's something wrong with the article
        try:
            articles.append(g.extract(url=l).cleaned_text.replace('\n',' '))
        except ParserError as e:
            articles.append('')
    if len(articles)%100 == 0:
        print(str(len(articles)) + " articles processed")
with open('articles.txt','w') as f:
    f.write('\n'.join(articles))
