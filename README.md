# Text Summary NLP

A powerful and flexible text summarization system that uses multiple algorithms to extract the most important sentences from texts. Supports both Turkish and English languages.

## Features

- 🤖 **Multiple Algorithms**: TF-IDF frequency-based, TextRank, position-based, and hybrid approaches
- 🌍 **Multi-language Support**: Turkish and English
- ⚡ **Fast Processing**: Efficient algorithms for quick summarization
- 🎯 **Customizable**: Adjustable summary length and algorithm selection
- 📊 **Statistics**: Get detailed statistics about the summarization process
- 🖥️ **Web Interface**: Built-in Streamlit web interface for easy use

## Installation

### Requirements

```bash
pip install numpy scikit-learn streamlit
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from textsummarynlp import MetinOzetleyici

# Initialize summarizer for Turkish
ozetleyici = MetinOzetleyici(dil='tr')

# Summarize text
text = """
Yapay zeka, bilgisayar biliminin en heyecan verici alanlarından biridir. 
Son yıllarda makine öğrenmesi ve derin öğrenme teknolojilerindeki gelişmeler 
sayesinde yapay zeka uygulamaları günlük hayatımızın her alanına girdi.
"""

ozet = ozetleyici.ozetle(text, ozet_orani=0.3, algoritma='hybrid')
print(ozet)
```

### English Example

```python
# Initialize summarizer for English
summarizer = MetinOzetleyici(dil='en')

text = """
Artificial intelligence is one of the most exciting fields in computer science. 
Thanks to recent advances in machine learning and deep learning technologies, 
AI applications have entered every aspect of our daily lives.
"""

summary = summarizer.ozetle(text, ozet_orani=0.3, algoritma='hybrid')
print(summary)
```

## Algorithms

The system supports four different summarization algorithms:

### 1. Frequency-based (`'frequency'`)
- Uses TF-IDF-like word frequency analysis
- Scores sentences based on important word frequencies
- Best for: Documents with clear keyword patterns

### 2. TextRank (`'textrank'`)
- Graph-based algorithm inspired by PageRank
- Considers sentence similarity relationships
- Best for: Documents with interconnected sentences

### 3. Position-based (`'position'`)
- Scores sentences based on their position in the text
- First 20% and last 10% get higher scores
- Best for: Well-structured documents (introduction/conclusion)

### 4. Hybrid (`'hybrid'`) - Recommended
- Combines all three algorithms with weights:
  - Frequency: 40%
  - TextRank: 40%
  - Position: 20%
- Best for: General purpose summarization

## API Reference

### `MetinOzetleyici`

#### Constructor

```python
MetinOzetleyici(dil: str = 'tr')
```

**Parameters:**
- `dil` (str): Language code - `'tr'` for Turkish, `'en'` for English

#### Methods

##### `ozetle()`

Summarizes the given text.

```python
ozetle(
    text: str,
    ozet_orani: float = 0.3,
    min_cumle: int = 3,
    max_cumle: int = 10,
    algoritma: str = 'hybrid'
) -> str
```

**Parameters:**
- `text` (str): Text to summarize
- `ozet_orani` (float): Summary ratio (0-1), default 0.3 (30%)
- `min_cumle` (int): Minimum number of sentences in summary, default 3
- `max_cumle` (int): Maximum number of sentences in summary, default 10
- `algoritma` (str): Algorithm to use - `'frequency'`, `'textrank'`, `'position'`, or `'hybrid'`

**Returns:**
- `str`: Summarized text

##### `batch_ozetle()`

Summarizes multiple texts in batch.

```python
batch_ozetle(texts: List[str], **kwargs) -> List[str]
```

**Parameters:**
- `texts` (List[str]): List of texts to summarize
- `**kwargs`: Same parameters as `ozetle()`

**Returns:**
- `List[str]`: List of summarized texts

##### `get_summary_stats()`

Returns statistics about the summarization.

```python
get_summary_stats(original_text: str, summary: str) -> Dict[str, any]
```

**Parameters:**
- `original_text` (str): Original text
- `summary` (str): Summary text

**Returns:**
- `Dict` with keys:
  - `orijinal_cumle_sayisi`: Number of sentences in original text
  - `ozet_cumle_sayisi`: Number of sentences in summary
  - `orijinal_kelime_sayisi`: Number of words in original text
  - `ozet_kelime_sayisi`: Number of words in summary
  - `sikistirma_orani`: Compression ratio (percentage)
  - `cumle_azalma_orani`: Sentence reduction ratio (percentage)

## Usage Examples

### Example 1: Basic Summarization

```python
from textsummarynlp import MetinOzetleyici

ozetleyici = MetinOzetleyici(dil='tr')
text = "Your long text here..."

ozet = ozetleyici.ozetle(text, ozet_orani=0.3)
print(ozet)
```

### Example 2: Custom Algorithm and Length

```python
ozetleyici = MetinOzetleyici(dil='en')

# Use TextRank algorithm
ozet = ozetleyici.ozetle(
    text, 
    ozet_orani=0.5,  # 50% summary
    algoritma='textrank',
    max_cumle=15
)
```

### Example 3: Batch Processing

```python
texts = [
    "First long text...",
    "Second long text...",
    "Third long text..."
]

ozetleyici = MetinOzetleyici(dil='tr')
summaries = ozetleyici.batch_ozetle(texts, ozet_orani=0.3)
```

### Example 4: Get Statistics

```python
ozetleyici = MetinOzetleyici(dil='tr')
ozet = ozetleyici.ozetle(text, ozet_orani=0.3)
stats = ozetleyici.get_summary_stats(text, ozet)

print(f"Original sentences: {stats['orijinal_cumle_sayisi']}")
print(f"Summary sentences: {stats['ozet_cumle_sayisi']}")
print(f"Compression: {stats['sikistirma_orani']}")
```

## Running the Demo

### Command Line Demo

```bash
python textsummarynlp.py
```

This will run a demo with example Turkish and English texts.

### Web Interface

```bash
python textsummarynlp.py --arayuz
```

This will start a Streamlit web interface at `http://localhost:8501` where you can:
- Paste or type your text
- Select language (Turkish/English)
- Adjust summary length
- View statistics
- See the summary instantly

## How It Works

1. **Text Preprocessing**: Cleans and normalizes the input text
2. **Sentence Splitting**: Divides text into sentences
3. **Word Tokenization**: Extracts and filters words (removes stop words)
4. **Scoring**: Calculates importance scores using selected algorithm(s)
5. **Selection**: Selects top-scoring sentences based on summary ratio
6. **Reordering**: Orders selected sentences in original text order
7. **Output**: Returns the final summary

## Algorithm Details

### Frequency-based Scoring
- Calculates word frequencies across all sentences
- Normalizes frequencies (0-1 scale)
- Scores sentences by average word frequency
- Sentences with more frequent important words score higher

### TextRank Algorithm
- Builds similarity matrix between sentences (Jaccard similarity)
- Normalizes similarity matrix
- Applies PageRank-like iterative scoring
- Sentences similar to other important sentences get higher scores

### Position-based Scoring
- First 20% of sentences: Score 1.0
- Last 10% of sentences: Score 0.6
- Middle sentences: Score 0.3
- Based on the observation that introductions and conclusions are often important

### Hybrid Approach
- Combines all three scoring methods
- Weights: Frequency (40%) + TextRank (40%) + Position (20%)
- Provides balanced results for various text types

## Tips for Best Results

1. **Text Length**: Works best with texts containing at least 5-6 sentences
2. **Language**: Make sure to select the correct language (`'tr'` or `'en'`)
3. **Summary Ratio**: 
   - 0.2-0.3: Very concise summary
   - 0.3-0.5: Balanced summary
   - 0.5-0.7: Detailed summary
4. **Algorithm Selection**:
   - Use `'hybrid'` for general purpose (recommended)
   - Use `'frequency'` for keyword-rich documents
   - Use `'textrank'` for documents with interconnected ideas
   - Use `'position'` for well-structured academic papers

## Limitations

- Works best with extractive summarization (selects existing sentences)
- Requires sufficient text length (minimum 3-5 sentences recommended)
- Performance depends on text structure and quality
- Stop words are language-specific and predefined

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Author

Text Summary NLP - Advanced Text Summarization System

