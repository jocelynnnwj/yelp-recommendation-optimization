# Yelp Rating Prediction - Project Documentation

## 📋 Project Overview

This project implements a hybrid recommender system for predicting user ratings on Yelp businesses. The system combines collaborative filtering (matrix factorization) with content-based features to achieve high prediction accuracy.

## 🎯 Problem Statement

**Task**: Predict user ratings (1-5 stars) for business-user pairs in the Yelp dataset.

**Challenge**: Build a recommendation system that can handle:
- Cold-start scenarios (new users/businesses)
- Sparse rating matrices
- Complex user-business interactions
- Large-scale data processing

## 📊 Dataset Description

### Official Competition Data
- **Dataset Link:** [Yelp Competition Datasets (Google Drive)](https://drive.google.com/drive/folders/1SIlY40owpVcGXJw3xeXk76afCwtSUx11?usp=sharing)
- **Data Split:**
  - 60% for training
  - 20% for validation
  - 20% for testing

### Programming Environment
- Python 3.6 (project is compatible with Python 3.6+)
- (Original competition also supported Scala 2.12, JDK 1.8, Spark 3.1.2)

### Yelp Open Dataset Components

**Training Data (`yelp_train.csv`)**:
- Format: CSV with columns `user_id`, `business_id`, `stars`
- Target variable: `stars` (1-5 rating scale)
- Contains historical user-business rating pairs

**Business Data (`business.json`)**:
- Business metadata in JSON format
- Fields: business_id, name, location, attributes, review statistics
- Key attributes: price range, credit card acceptance, accessibility, service features

**User Data (`user.json`)**:
- User profile information in JSON format
- Fields: user_id, review count, friends, engagement metrics, elite status
- Social features: compliments, useful/funny/cool votes, fans

#### Available Data Files

- **yelp_train.csv**: Training data (`user_id`, `business_id`, `stars`)
- **yelp_val.csv**: Validation data (same format as training)
- **review_train.json**: Review text and metadata for training pairs
- **user.json**: All user metadata
- **business.json**: All business metadata (locations, attributes, categories)
- **checkin.json**: User checkins for businesses
- **tip.json**: Tips (short reviews) written by users
- **photo.json**: Photo data, captions, and classifications

*Note: The test dataset is not shared by the competition organizers.*

*This project primarily uses `yelp_train.csv`, `user.json`, and `business.json` for feature engineering and model training. The other files are available for advanced feature engineering or further research.*

## 🏗️ System Architecture

### Hybrid Approach
1. **Matrix Factorization (Collaborative Filtering)**
   - Custom implementation with SGD optimization
   - Learns user and business latent embeddings
   - Handles bias terms for users and businesses

2. **XGBoost (Content-Based Features)**
   - 27 engineered features from business and user attributes
   - Handles cold-start scenarios
   - Models complex nonlinear relationships

3. **Ensemble Blending**
   - Weighted combination: 11% MF + 89% XGBoost
   - Optimized for best validation RMSE

## 🔧 Technical Implementation

### Matrix Factorization Model
```python
class MFRecommender:
    - Latent dimensions: 18
    - Training steps: 18
    - Learning rate: 0.015
    - Regularization: 0.04
    - Initialization: Normal(0, 0.15)
```

### XGBoost Model
```python
XGBRegressor:
    - Learning rate: 0.045
    - Max depth: 5
    - Estimators: 850
    - Random state: 1234
```

### Feature Engineering (27 Features)
**Business Features (10)**:
- Rating statistics (stars, review_count)
- Geographic location (latitude, longitude)
- Business attributes (price_range, credit_card, etc.)
- Service features (reservations, table_service, etc.)

**User Features (17)**:
- Profile statistics (review_count, friends, fans)
- Engagement metrics (useful, funny, cool)
- Elite status and average rating
- Compliment categories (9 different types)

## 📈 Performance Results

### Competition Performance
- **Ranking**: Top 20% 🏅
- **RMSE**: 0.977853
- **Execution Time**: 800.58 seconds

### Error Distribution
- 0-1 stars: 102,256 predictions (85.7%)
- 1-2 stars: 32,831 predictions (11.0%)
- 2-3 stars: 6,149 predictions (2.6%)
- 3-4 stars: 808 predictions (0.7%)
- 4+ stars: 0 predictions

## 🚀 Key Innovations

1. **Custom Matrix Factorization**: Built from scratch with optimized SGD
2. **Intelligent Feature Blending**: Optimal ensemble weights for maximum performance
3. **Robust Data Handling**: Smart fallback strategies for missing values
4. **Performance Optimization**: Efficient data structures and algorithms
5. **Reproducible Results**: Consistent random seeds and deterministic execution

## 💻 Technical Skills Demonstrated

### Programming & Tools
- **Python**: Core implementation with OOP design
- **PySpark**: Large-scale distributed data processing
- **NumPy/Pandas**: Efficient numerical operations and data manipulation
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **JSON**: Complex nested data parsing

### Machine Learning
- **Collaborative Filtering**: Custom matrix factorization implementation
- **Ensemble Methods**: Intelligent model blending strategies
- **Feature Engineering**: Domain-specific feature extraction
- **Hyperparameter Optimization**: Performance vs. speed tuning
- **Model Evaluation**: Comprehensive error analysis

### Data Science
- **Big Data Processing**: Scalable Spark implementation
- **Statistical Analysis**: Detailed error distribution analysis
- **Data Preprocessing**: Robust missing value handling
- **Performance Optimization**: Efficient algorithms and memory management

## 🔬 Research Contributions

- Demonstrated effectiveness of hybrid approaches in recommendation systems
- Validated ensemble methods for improving prediction accuracy
- Showed importance of feature engineering in cold-start scenarios
- Provided insights into optimal hyperparameter selection
- Proved approach effectiveness against diverse ML solutions

## 📁 Project Structure

```
├── README.md                 # Main project documentation
├── PROJECT_DOCUMENTATION.md  # Detailed technical documentation
├── yelp_recommender.py       # Main implementation
├── requirements.txt          # Python dependencies
├── sample_predictions.csv    # Example output
└── .gitignore               # Git configuration
```

## 🎓 Learning Outcomes

- Advanced machine learning algorithm implementation
- Large-scale data processing with PySpark
- Feature engineering for recommendation systems
- Model ensemble techniques and optimization
- Production-ready code with error handling
- Competitive machine learning problem-solving

---

*This documentation captures the complete project scope, technical implementation, and competitive achievements of the Yelp rating prediction system.* 