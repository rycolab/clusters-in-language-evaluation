import sys
import time
from datasets import load_dataset
author_dataset = load_dataset('Fraser/news-category-dataset', split='train')
from goose3 import Goose
from lxml.etree import ParserError
g = Goose()

interval = int(sys.argv[1])
from collections import Counter
author_labels = [x.split(",")[0] for x in author_dataset['authors']]
counts = sorted(Counter(author_labels).items(), key=lambda item: item[1], reverse=True)

authors = set([a for a, n in counts if n >=400])
author_dataset = load_dataset('Fraser/news-category-dataset', split='test')
author_labels = [x.split(",")[0] for x in author_dataset['authors']]
author_dataset_filtered = [(author, article) for author, article in zip(author_labels, author_dataset['link']) if (author and author in authors)]

author_dataset_filtered = author_dataset_filtered[(interval*1000):(interval + 1)*1000]
articles = []
for _, l in author_dataset_filtered:
    try:
        articles.append(g.extract(url=l).cleaned_text.replace('\n',' '))
    except ParserError as e:
        time.sleep(65)
        try:
            articles.append(g.extract(url=l).cleaned_text.replace('\n',' '))
        except ParserError as e:
            print(e)
            articles.append('')
    if len(articles)%100 == 0:
        print(len(articles))
with open('articles_test_'+ str(interval)+'.txt','w') as f:
    f.write('\n'.join(articles))
