# spam-message-detector
📩 ML-powered spam detector | Classifies messages as SPAM/NOT SPAM | Python + Scikit-learn + NLTK


# 📩 Spam Message Classifier

![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning model that identifies spam messages with **98% accuracy**.

## ✨ Features
- Instant SPAM/NOT SPAM classification
- Pre-trained model included
- Simple console interface
- Custom training support

## 🚀 Quick Start
```bash
# 1. Clone repo
git clone https://github.com/Dharani-S93/spam-message-detector.git

# 2. Install requirements
pip install -r requirements.txt

# 3. Run detector
python spam_detector.py
```

## 📸 Demo
```
Enter message: "Claim your free prize now!"
Result: SPAM

Enter message: "Let's meet for coffee"
Result: NOT SPAM
```

## 📂 Project Structure
```
spam-message-detector/
├── spam_detector.py       # Main classifier
├── requirements.txt       # Dependencies
├── data/spam.csv          # Sample dataset
└── model/                 # Saved ML models
```

## 📊 Performance
| Metric     | Score |
|------------|-------|
| Accuracy   | 98.2% |
| Precision  | 99.1% |
| Recall     | 96.7% |

## 🤝 Contribute
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
