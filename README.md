# Document Retrieval System

An efficient, lightweight document retrieval system that supports multiple query types and ranking algorithms, specifically designed for processing TSV-formatted document collections.

## Key Features

- **Multiple Query Types**:
  - Natural Language Queries - Support for complete sentences and phrases or keywords
  - Keyword Queries - Documents scored using TF-IDF or BM25 algorithms
  - Boolean Queries - Supports AND, OR, NOT operators and nested expressions with parentheses
  - Wildcard Queries - Supports * wildcard for pattern matching

- **Advanced Ranking Algorithms**:
  - TF-IDF Cosine Similarity - Traditional vector space model, suitable for general text similarity calculation
  - BM25 Ranking - Modern probabilistic model considering term frequency saturation and document length normalization
  - Hybrid Mode - Uses TF-IDF to filter candidate documents, then rerankings with BM25, balancing efficiency and accuracy

- **Intelligent Text Processing**:
  - Automatic tokenization and stopword removal
  - Word lemmatization to improve matching (e.g., "running" → "run")
  - Case-insensitive matching

- **TSV File Support**:
  - Designed for processing large TSV document collections like MS MARCO
  - Automatically parses document ID, URL, title, and content fields from TSV files
  - Supports batch processing of multiple TSV files

## Installation and Dependencies

```bash
# Install dependencies
pip install nltk tqdm

# Download NLTK data (optional but recommended for improved processing)
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
```

## Document Format Requirements

The system is designed to process TSV format files. Each line should follow this format:

```
DocumentID<TAB>URL<TAB>Title<TAB>Content
```

Example:
```
D1<TAB>http://example.com/1<TAB>Example Document 1<TAB>This is the content of the first document...
D2<TAB>http://example.com/2<TAB>Example Document 2<TAB>This is the content of another example document...
```

All TSV files should be placed in a designated directory. The system will load all `.tsv` files from that directory. The system automatically indexes and processes all files without manual segmentation or preprocessing.

## Usage

### Basic Execution

```bash
python homework.py --doc_dir docs_data
```

This will start the interactive retrieval system, loading all TSV files from the specified directory (`docs_data`). Initial loading may take some time for index building.

### Command Line Arguments

```bash
python homework.py --doc_dir <document_directory>
```

- `--doc_dir`: Required, specifies the directory containing TSV documents

### Interactive Query Mode

After startup, the program enters interactive mode where you can:

1. Enter your query (keywords, sentence, boolean expression, or wildcard pattern)
   ```
   Enter query: What are the recent advances in artificial intelligence for healthcare
   ```

2. Specify the number of results to return (K value)
   ```
   Number of results K: 5
   ```

3. Choose the search mode (TF-IDF or Hybrid mode)
   ```
   Please choose the search mode (1 or 2): 2
   ```
   - Option `1`: Traditional TF-IDF
   - Option `2`: Hybrid mode (TF-IDF filtering + BM25 reranking)

4. View matching documents and their relevance scores in JSON format, including:
   - Document ID
   - Rank
   - Relevance score
   - Document content

5. Type `exit` to quit the program



### Query Processing Flow

1. The system first analyzes the query type (natural language, keyword, boolean, wildcard)
2. For natural language and keyword queries:
   - Tokenizes the input text
   - Removes stopwords
   - Applies lemmatization
   - Creates a weighted term vector
3. Selects the appropriate processing method based on query type
4. Scores and ranks matching documents
5. Finally returns the sorted results, ranked by relevance score from highest to lowest

### Example Output

```json
{
  "k": 2,
  "docs": [
    {
      "rank": 1,
      "document_id": "D123",
      "score": 0.856,
      "content": "This is a document about artificial intelligence..."
    },
    {
      "rank": 2,
      "document_id": "D456",
      "score": 0.723,
      "content": "Deep learning is a branch of artificial intelligence..."
    }
  ]
}
```



