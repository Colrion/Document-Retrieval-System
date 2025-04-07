import os, re, json, math, argparse, csv, time
from collections import Counter, defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Set CSV field size limit
csv.field_size_limit(1000000000)


# Ensure NLP resources are available
for resource in ['punkt', 'stopwords', 'wordnet']:
    try: nltk.data.find(f'{"tokenizers/" if resource=="punkt" else "corpora/"}{resource}')
    except: nltk.download(resource)

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# Constants
DEFAULT_TOP_K = 5
HYBRID_MULTIPLIER = 4  # Hybrid search multiplier

class IRSystem:
    """Information Retrieval System"""
    
    def __init__(self):
        # Core data structures
        self.documents = {}                       # Document content
        self.index = defaultdict(set)             # Inverted index
        self.term_frequency = defaultdict(dict)   # Term frequency
        self.inverse_doc_frequency = {}           # Inverse document frequency
        self.document_lengths = {}                # Document lengths
        self.document_vectors = defaultdict(dict) # Document vectors 
        self.metadata = {}                        # Metadata
        self.average_length = 0                   # Average length
        self.k1, self.b = 1.5, 0.75               # BM25 parameters

    def tokenize(self, text):
        """Text tokenization processing"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Filter and lemmatize
        processed = []
        for t in tokens:
            if t.isalnum() and (len(t) > 2 or t not in STOPWORDS):
                if t.isalpha():
                    t = LEMMATIZER.lemmatize(t)
                processed.append(t)
        return processed

    def load(self, directory_path):
        """Load document collection"""
        if not os.path.exists(directory_path): 
            raise FileNotFoundError(f"Document directory does not exist: {directory_path}")
        
        # Get TSV file list and process
        tsv_files = [f for f in os.listdir(directory_path) if f.endswith('.tsv')]

        if not tsv_files:
            print("No TSV files found")
            return 0
        
        document_count = 0
        print(f"Found {len(tsv_files)} TSV files")
        
        # Process each file
        for filename in tqdm(tsv_files, desc="Loading documents"):
            try:
                # Read and index documents
                documents = {}
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as tsv_file:
                    for row in csv.reader(tsv_file, delimiter='\t'):
                        if len(row) >= 4:
                            documents[row[0]] = {"url": row[1], "title": row[2], "body": row[3]}
                
                # inverted index
                for document_id, metadata in tqdm(documents.items(), desc="Indexing documents"):
                    # Process document content
                    text = metadata["body"]
                    tokens = self.tokenize(text)
                    term_counts = Counter(tokens)
                    
                    # Update index
                    self.documents[document_id] = text
                    self.document_lengths[document_id] = len(tokens)
                    self.metadata[document_id] = metadata
                    
                    for term, count in term_counts.items():
                        self.index[term].add(document_id)
                        self.term_frequency[term][document_id] = count
                    
                    document_count += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
        # Build index statistics
        if document_count > 0:
            print("Building index statistics...")
            # Calculate average length and IDF
            self.average_length = sum(self.document_lengths.values()) / max(1, len(self.documents))
            num_docs = len(self.documents)
            self.inverse_doc_frequency = {term: math.log(num_docs / len(docs)) 
                                         for term, docs in self.index.items()}
            
            # Calculate vectors
            for term, doc_freqs in tqdm(self.term_frequency.items(), desc="Building vectors"):
                idf = self.inverse_doc_frequency[term]
                for doc_id, freq in doc_freqs.items():
                    self.document_vectors[doc_id][term] = (freq / self.document_lengths[doc_id]) * idf
            
            print(f"Processing complete, {document_count} documents total")
        
        return document_count
    
    def search(self, query, k=DEFAULT_TOP_K, use_hybrid=False):
        """Search documents"""
        query = query.strip()
        if not query: return {"k": k, "docs": []}
        
        try:
            # Preprocess query and score
            matching_documents, query_tokens = self._preprocess_query(query)
            if not query_tokens or not matching_documents:
                return {"k": k, "docs": []}
            
            # Search scoring
            if use_hybrid:
                # Hybrid search: TF-IDF filtering + BM25 reranking
                tfidf_scores = self._score(query_tokens, matching_documents, 'tfidf')
                if tfidf_scores:
                    top_k = min(k * HYBRID_MULTIPLIER, len(matching_documents))
                    top_docs = {doc_id for doc_id, _ in sorted(tfidf_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:top_k]}
                    scores = self._score(query_tokens, top_docs, 'bm25')
                else:
                    scores = {}
            else:
                # Standard TF-IDF
                scores = self._score(query_tokens, matching_documents, 'tfidf')
            
            # Return results
            return self._format_results(scores, k)
        except Exception as e:
            print(f"Search error: {e}")
            return {"k": k, "docs": []}
    
    def _preprocess_query(self, query):
        """Preprocess query"""

        # wildcard query
        if '*' in query:
            return self._process_wildcard(query)
        # boolean query
        elif any(op in query for op in [' AND ', ' OR ', 'NOT ']):
            matching_docs = self._boolean_search(query)
            query_tokens = self.tokenize(re.sub(r'\bAND\b|\bOR\b|\bNOT\b|\(|\)', ' ', query))
            return matching_docs, query_tokens
        # term query
        else:
            query_tokens = self.tokenize(query)
            if not query_tokens:
                query_tokens = [w.lower() for w in word_tokenize(query) if w.isalnum()]
            return set(self.documents.keys()), query_tokens
    
    def _process_wildcard(self, query):
        """Process wildcard query"""
        # Match terms and documents
        # Find all wildcard patterns
        patterns = [token for token in query.lower().split() if '*' in token]

        # Expand patterns to actual terms in the index
        matched_terms = set()
        for pattern in patterns:
            regex = f"^{pattern.replace('*', '.*')}$"
            matched_terms.update(t for t in self.index if re.match(regex, t))

        # Get documents that contain any of the matched terms
        matched_docs = set()
        for term in matched_terms:
            matched_docs.update(self.index[term])

        # Prepare tokens for ranking
        tokens = self.tokenize(query.replace('*', ''))
        tokens.extend(matched_terms)

        return matched_docs, tokens
    
    def _boolean_search(self, query):
        """Boolean query processing"""
        def parse(q):
            """Parse boolean expression into an abstract syntax tree"""
            q = q.strip()

            # Remove outer parentheses if the entire query is wrapped
            if q.startswith('(') and q.endswith(')'):
                return parse(q[1:-1].strip())

            # Handle NOT operation
            if q.startswith('NOT '):
                sub_expr = q[4:].strip()
                return {'op': 'NOT', 'expr': parse(sub_expr)}

            # Handle AND operation
            if ' AND ' in q:
                left, right = q.split(' AND ', 1)
                return {
                    'op': 'AND',
                    'left': parse(left.strip()),
                    'right': parse(right.strip())
                }

            # Handle OR operation
            if ' OR ' in q:
                left, right = q.split(' OR ', 1)
                return {
                    'op': 'OR',
                    'left': parse(left.strip()),
                    'right': parse(right.strip())
                }

            # Handle wildcard search
            if '*' in q:
                return {'op': 'WILDCARD', 'pattern': q}

            # Handle single term
            return {'op': 'TERM', 'term': q}    
        
        def execute(tree):
            """Execute expression"""
            op = tree['op']
            if op == 'AND': return execute(tree['left']) & execute(tree['right'])
            elif op == 'OR': return execute(tree['left']) | execute(tree['right'])
            elif op == 'NOT': return set(self.documents.keys()) - execute(tree['expr'])
            elif op == 'TERM':
                term = tree['term'].lower()
                try: term = LEMMATIZER.lemmatize(term) 
                except: pass
                return self.index.get(term, set())
            elif op == 'WILDCARD':
                pattern = tree['pattern'].lower().replace('*', '.*')
                terms = {t for t in self.index if re.match(f"^{pattern}$", t)}
                docs = set()
                for t in terms: docs.update(self.index[t])
                return docs
            return set()
        
        return execute(parse(query))
    
    def _score(self, query_tokens, document_ids, mode='tfidf'):
        """Score calculation (tfidf or bm25)"""
        if mode == 'tfidf':
            # TF-IDF cosine similarity
            counts = Counter(query_tokens)
            query_len = max(len(query_tokens), 1)
            query_vector = {term: (count/query_len) * self.inverse_doc_frequency.get(term, 0)
                           for term, count in counts.items() 
                           if term in self.inverse_doc_frequency}
            
            if not query_vector: return {}
            
            # Similarity calculation
            scores = {}
            for doc_id in document_ids:
                doc_vector = self.document_vectors[doc_id]
                common_terms = set(query_vector) & set(doc_vector)
                if not common_terms: continue
                
                # Dot product and norms
                dot_product = sum(query_vector[t] * doc_vector[t] for t in common_terms)
                query_norm = math.sqrt(sum(v**2 for v in query_vector.values()))
                doc_norm = math.sqrt(sum(val**2 for val in doc_vector.values()))
                
                # Similarity
                similarity = dot_product / (query_norm * doc_norm) if query_norm * doc_norm > 0 else 0
                if similarity > 0: scores[doc_id] = similarity
            
            return scores
            
        elif mode == 'bm25':
            # BM25 scoring
            scores = defaultdict(float)
            for term in set(query_tokens):
                if term not in self.index: continue
                
                relevant_docs = self.index[term] & document_ids
                if not relevant_docs: continue
                
                idf = self.inverse_doc_frequency.get(term, 0)
                for doc_id in relevant_docs:
                    tf = self.term_frequency[term][doc_id]
                    doc_len = self.document_lengths[doc_id]
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.average_length)
                    scores[doc_id] += idf * (numerator / denominator)
            
            return dict(scores)
        return {}
    
    def _format_results(self, scores, k):
        """Format search results"""
        results = []
        for rank, (doc_id, score) in enumerate(sorted(scores.items(), 
                                              key=lambda x: x[1], reverse=True)[:k], 1):
            # Build result
            result = {"rank": rank, "document_id": doc_id, "score": score, 
                     "content": self.documents[doc_id]}
            
            
            results.append(result)
        
        return {"k": k, "docs": results}

def main():
    """Main function"""
    # Parse arguments and initialize
    parser = argparse.ArgumentParser(description='Information Retrieval System')
    parser.add_argument('--doc_dir', type=str, required=True, help='Document directory path')
    ir_system = IRSystem()
    
    try:
        # Load documents
        print("\n===== Loading Documents =====")
        args = parser.parse_args()
        start_time = time.time()
        doc_count = ir_system.load(args.doc_dir)
        print(f"Successfully loaded {doc_count} documents, took {time.time()-start_time:.2f} seconds")
        
        if doc_count == 0: 
            return print("No available documents, program exit")
        
        # Interactive mode
        print("\n===== Document Retrieval System =====\nType 'exit' to quit\n" + "-" * 30)
        
        while True:
            # Execute query
            query = input("\nEnter query: ").strip()
            if not query or query.lower() == 'exit': 
                break
            
            try: 
                k = max(1, int(input("Number of results K: ")))
            except: 
                k = DEFAULT_TOP_K
            
            print("""\nSearch mode:
1: TF-IDF (default)
2: Hybrid (TF-IDF for ranking, BM25 for reranking)
                  """)
            
            # Get search mode
            use_hybrid = input("Please choose the search mode (1 or 2): ") == '2'
            

            # Search and display results
            start_time = time.time()
            results = ir_system.search(query, k, use_hybrid)
            search_time = time.time() - start_time
            
            if not results["docs"]:
                print("No matching documents found")
            else:
                print(f"\nFound {len(results['docs'])} matching documents, took {search_time:.4f} seconds")
                print(f"Search mode: {'TF-IDF' if not use_hybrid else 'Hybrid mode (TF-IDF+BM25)'}")
                print(json.dumps(results, ensure_ascii=False, indent=2))
                
    except Exception as e:
        print(f"Program execution error: {e}")

if __name__ == "__main__":
    main()
    
