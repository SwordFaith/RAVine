''' Tool for matching docids and web urls in a web corpus '''

import os
import json


class UrlDocMapper:
    
    def __init__(self,
                 corpus_name: str,
                 mapper_path: str,
                 corpus_path: str=None):
        
        self.corpus_name = corpus_name
        self.mapper_path = mapper_path
        self.corpus_path = corpus_path # `corpus_path` cannot be None if the mapper file needs to be initialized

        if os.path.isfile(self.mapper_path):
            self._load_mapper()
        else:
            self._generate_mapper()
    
    def _load_mapper(self):
        with open(self.mapper_path, 'r') as file:
            self.mapper = json.load(file)
    
    def _generate_mapper(self):
        self.mapper = dict()
        with open(self.corpus_path, 'r') as file:
            for line in file:
                doc = json.loads(line)
                self.mapper[doc['url']] = doc['docid']
        
        with open(self.mapper_path, 'w') as file:
            json.dump(self.mapper, file)

    def get_docid_by_url(self, url):
        return self.mapper.get(url, None)
