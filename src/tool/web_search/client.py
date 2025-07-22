''' search client (sparse and dense) '''

from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from abc import ABC, abstractmethod
from .url2doc import UrlDocMapper
import json


class SearchClient(ABC):

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def fetch(self):
        pass



class BM25Client(SearchClient):

    def __init__(self,
                 corpus_name: str,
                 index_path: str,
                 mapper_path: str,
                 corpus_path: str=None,
                 k1: float=0.9,
                 b: float=0.4):
        
        self.corpus_name = corpus_name
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.mapper_path = mapper_path

        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)

        self.mapper = UrlDocMapper(corpus_name, mapper_path, corpus_path)
    
    def search(self, query, topk=10):
        hits = self.searcher.search(query, k=topk)

        res = []
        for i in range(len(hits)):
            docid = hits[i].docid
            doc = json.loads(self.searcher.doc(docid).raw())
            res.append({
                'title': doc['title'],
                'headings': doc['headings'],
                'url': doc['url'],
                # 'score': hits[i].score.item(),
            })
        
        return res
    
    def fetch(self, url, format=None):
        try:
            docid = self.mapper.get_docid_by_url(url)
        except KeyError:
            print(f'Fetch Error: Wrong URL - {url}')
            return None
        
        doc = json.loads(self.searcher.doc(docid).raw())
        return doc['body']



class DenseClient(SearchClient):

    def __init__(self,
                 embedder_name: str,
                 corpus_name: str,
                 index_path: str,
                 mapper_path: str,
                 lucene_index_path: str,
                 corpus_path: str=None):
        
        self.embedder_name = embedder_name
        self.corpus_name = corpus_name
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.mapper_path = mapper_path

        self.searcher = FaissSearcher(index_path, embedder_name)

        # load lucene index to access to the raw text of the corpus
        self.lucene_index_path = lucene_index_path
        self.lucene_searcher = LuceneSearcher(lucene_index_path)

        self.mapper = UrlDocMapper(corpus_name, mapper_path, corpus_path)
    
    def search(self, query, topk=10,
               instruction='Given a web search query, retrieve relevant webs that answer the query'):
        if instruction:
            query = f'Instruction: {instruction}\nQuery: {query}'
        
        hits = self.searcher.search(query, k=topk)

        res = []
        for i in range(len(hits)):
            docid = hits[i].docid
            # faiss index does not provide access to the raw text of the doc
            doc = json.loads(self.lucene_searcher.doc(docid).raw())
            res.append({
                'title': doc['title'],
                'headings': doc['headings'],
                'url': doc['url'],
                # 'score': hits[i].score.item(),
            })
        
        return res
    
    def fetch(self, url, format=None):
        try:
            docid = self.mapper.get_docid_by_url(url)
        except KeyError:
            print(f'Fetch Error: Wrong URL - {url}')
            return None
        
        doc = json.loads(self.lucene_searcher.doc(docid).raw())
        return doc['body']


