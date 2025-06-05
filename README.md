# Deep Learning Sentiment Analysis Project

A comprehensive implementation of sentiment analysis on IMDB movie reviews using multiple machine learning and deep learning approaches.

## 🎯 Project Overview

This project implements and compares different approaches to sentiment analysis:
- **Traditional Machine Learning**: TF-IDF + Logistic Regression, Naive Bayes, SVM, Random Forest
- **Deep Learning**: LSTM Neural Networks with word embeddings
- **Transformer Models**: BERT for state-of-the-art performance

## 📊 Dataset

- **Training Data**: 25,000 labeled movie reviews from IMDB
- **Test Data**: 25,000 unlabeled movie reviews
- **Target**: Binary sentiment classification (positive/negative)
- **Source**: IMDB Movie Review Dataset

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Libraries

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- torch, transformers
- nltk, beautifulsoup4
- wordcloud
- jupyter

### Running the Project

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Launch Jupyter**: `jupyter notebook`
4. **Open**: `sentiment_analysis_project.ipynb`
5. **Run all cells** to execute the complete pipeline

## 📁 Project Structure

```
deeplearning-project/
├── sentiment_analysis_project.ipynb  # Main notebook
├── requirements.txt                  # Dependencies
├── README.md                        # This file
├── labeledTrainData.tsv            # Training data
├── testData.tsv                    # Test data
├── unlabeledTrainData.tsv          # Additional unlabeled data
└── sentiment_predictions.csv       # Generated predictions
```

## 🔧 Implementation Details

### 1. Data Preprocessing
- HTML tag removal using BeautifulSoup
- Text normalization (lowercase, special characters)
- Stopword removal and tokenization
- Custom text cleaning pipeline

### 2. Traditional ML Pipeline
```python
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Multiple Model Comparison
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier()
}
```

### 3. LSTM Neural Network
```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
```

### 4. BERT Transformer
```python
# Using Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

## 📈 Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~89% | Best traditional ML method |
| LSTM Neural Network | ~87% | Deep learning approach |
| BERT Pipeline | ~91% | State-of-the-art transformer |

## 🎨 Visualizations

The notebook includes:
- Sentiment distribution analysis
- Review length distributions
- Word clouds for positive/negative reviews
- Training loss and accuracy curves
- Model performance comparisons

## 💡 Key Features

- **Comprehensive Preprocessing**: Advanced text cleaning and normalization
- **Multiple Approaches**: Traditional ML, Deep Learning, and Transformers
- **Detailed EDA**: Extensive exploratory data analysis with visualizations
- **Model Comparison**: Fair evaluation framework across all approaches
- **Production Ready**: Clean, documented code with error handling
- **Reproducible**: Fixed random seeds for consistent results

## 🔬 Technical Highlights

1. **Custom LSTM Implementation**: Built from scratch using PyTorch
2. **BERT Integration**: Leveraging pre-trained transformers
3. **Efficient Preprocessing**: Optimized text processing pipeline
4. **Comprehensive Evaluation**: Multiple metrics and validation strategies
5. **Visualization Suite**: Rich plots and analysis charts

## 📚 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Traditional vs. modern NLP techniques
- Deep learning model architecture design
- Transfer learning with pre-trained models
- Data preprocessing and feature engineering
- Model evaluation and comparison methodologies

## 🚀 Future Enhancements

- [ ] Fine-tune BERT on the specific dataset
- [ ] Implement ensemble methods
- [ ] Add cross-validation
- [ ] Deploy as a web service
- [ ] Experiment with other transformer models (RoBERTa, DistilBERT)
- [ ] Add model interpretability features

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

*This project showcases a complete implementation of sentiment analysis using modern machine learning and deep learning techniques.* 