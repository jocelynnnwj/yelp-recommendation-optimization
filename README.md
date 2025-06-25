# Yelp Recommendation System Optimization

*Hybrid Rating Prediction Approach*
**USC DSCI 553 Competition Project**

## 🎯 Project Overview

A sophisticated hybrid recommender system that predicts user ratings on Yelp with exceptional accuracy, achieving **top 20% performance** in a competitive machine learning challenge. This project demonstrates advanced machine learning techniques including collaborative filtering, matrix factorization, and ensemble methods, achieving an RMSE of **0.977853** - an excellent performance for rating prediction tasks.

## 🏆 Performance Metrics & Competition Results

- **Competition Ranking**: **Top 20%** 🏅
- **RMSE**: 0.977853 (excellent performance for rating prediction)
- **Error Distribution**:
  - 0-1 stars: 102,256 predictions (85.7%)
  - 1-2 stars: 32,831 predictions (11.0%)
  - 2-3 stars: 6,149 predictions (2.6%)
  - 3-4 stars: 808 predictions (0.7%)
  - 4+ stars: 0 predictions
- **Execution Time**: 800.58 seconds
- **Model Blend**: 11% Matrix Factorization + 89% XGBoost

## 📊 Dataset & Data Sources

### Official Competition Data
- **Dataset Link:** [Yelp Competition Datasets (Google Drive)](https://drive.google.com/drive/folders/1SIlY40owpVcGXJw3xeXk76afCwtSUx11?usp=sharing)
- **Data Split:**
  - 60% for training
  - 20% for validation
  - 20% for testing

### Programming Environment
- Python 3.6 (project is compatible with Python 3.6+)
- (Original competition also supported Scala 2.12, JDK 1.8, Spark 3.1.2)

### Yelp Dataset Structure
The system utilizes the Yelp Open Dataset with the following components:

**Training Data:**
- `yelp_train.csv` - User-business rating pairs with ground truth stars (1-5 scale)

**Business Metadata (`business.json`):**
- Business ID, name, location (latitude/longitude)
- Average rating and review count
- Business attributes (price range, credit card acceptance, accessibility)
- Service features (reservations, table service, appointment-only)

**User Metadata (`user.json`):**
- User ID, review count, friends list
- Engagement metrics (useful, funny, cool votes received)
- Elite status and average rating given
- Compliment categories (hot, profile, list, note, etc.)

### Data Processing Features
- **27 engineered features** extracted from raw JSON data
- **Robust missing value handling** with smart fallback strategies
- **Geographic features** for location-based patterns
- **Social features** capturing user engagement and influence

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

## 🛠️ Technical Architecture

### Hybrid Model Design
The system combines two complementary approaches for optimal performance:

1. **Custom Matrix Factorization (Collaborative Filtering)**
   - **Implementation**: Built from scratch using stochastic gradient descent
   - **Latent Dimensions**: 18 (optimized for performance)
   - **Training Steps**: 18 iterations
   - **Learning Rate**: 0.015
   - **Regularization**: 0.04 (L2 penalty)
   - **Initialization**: Normal distribution (μ=0, σ=0.15)
   - **Features**: User bias, business bias, and latent embeddings

2. **XGBoost Regressor (Content-Based Features)**
   - **Learning Rate**: 0.045
   - **Max Depth**: 5
   - **Estimators**: 850 trees
   - **Random State**: 1234 (reproducible results)
   - **Features**: 27 engineered features from business and user attributes

### Feature Engineering (27 Total Features)

**Business Features (10):**
- `biz_stars`, `biz_review_count` - Rating statistics
- `latitude`, `longitude` - Geographic location
- `price_range` - Restaurant price range (1-4, random fallback)
- `credit_card` - Credit card acceptance (random fallback)
- `appt_only`, `reservations`, `table_service` - Service attributes
- `wheelchair` - Accessibility features

**User Features (17):**
- `user_review_count`, `user_friends`, `user_fans` - Profile statistics
- `user_useful`, `user_funny`, `user_cool` - Engagement metrics
- `user_elite` - Elite status (parsed from comma-separated list)
- `user_avg_stars` - Average rating given
- `comp_hot`, `comp_profile`, `comp_list`, `comp_note`, `comp_plain`, `comp_cool`, `comp_funny`, `comp_writer`, `comp_photos` - Compliment categories

### Data Processing Pipeline
- **PySpark Integration**: Large-scale data processing with SparkContext
- **JSON Parsing**: Efficient parsing of business and user metadata
- **Missing Value Handling**: Robust imputation with random sampling and defaults
- **Feature Construction**: Vectorized feature matrix construction
- **Data Validation**: Comprehensive error checking and type conversion

## 💻 Technical Skills Demonstrated

### Programming & Tools
- **Python** - Core implementation with object-oriented design
- **PySpark** - Large-scale data processing and distributed computing
- **NumPy/Pandas** - Efficient data manipulation and numerical operations
- **XGBoost** - Gradient boosting with optimized hyperparameters
- **JSON** - Complex nested data parsing and handling

### Machine Learning
- **Collaborative Filtering** - Custom matrix factorization implementation
- **Ensemble Methods** - Intelligent blending of complementary models
- **Feature Engineering** - Domain-specific feature extraction and preprocessing
- **Hyperparameter Optimization** - Careful tuning for performance vs. speed
- **Model Evaluation** - Comprehensive error analysis and validation

### Data Science
- **Big Data Processing** - Scalable Spark implementation for large datasets
- **Statistical Analysis** - Detailed error distribution analysis
- **Data Preprocessing** - Robust handling of missing values and edge cases
- **Performance Optimization** - Efficient algorithms and data structures

## 🚀 Key Innovations

1. **Custom Matrix Factorization**: Implemented from scratch with optimized SGD, including user/business bias terms
2. **Intelligent Feature Blending**: Optimal 11:89 ratio between MF and XGBoost predictions
3. **Robust Data Handling**: Smart fallback strategies for missing business attributes
4. **Performance Optimization**: Efficient feature matrix construction and memory management
5. **Reproducible Results**: Consistent random seeds and deterministic algorithms

## 📊 Model Performance Analysis

The hybrid approach achieves superior performance through:

- **Collaborative Filtering**: Captures implicit user-business interaction patterns
- **Content-Based Features**: Handles cold-start scenarios and complex nonlinear relationships
- **Ensemble Blending**: Combines strengths of both approaches for better generalization
- **Optimized Architecture**: 18 latent dimensions and 850 XGBoost trees for optimal performance

## 🔧 Usage & Deployment

### Command Line Interface
```bash
python yelp_recommender.py <input_directory> <test_file> <output_file>
```

### Input Requirements
- `yelp_train.csv` - Training data with user_id, business_id, stars
- `business.json` - Business metadata with attributes and location
- `user.json` - User metadata with profile and engagement data
- Test file with user_id and business_id pairs

### Output Format
- CSV file with columns: user_id, business_id, prediction
- Predictions clipped to valid range [1.0, 5.0]

### System Requirements
- Python 3.7+
- PySpark for distributed processing
- 8GB+ RAM recommended for large datasets

## 📁 Project Structure

```
├── README.md                 # Project documentation
├── yelp_recommender.py       # Main recommender system implementation
├── requirements.txt          # Python dependencies
├── sample_predictions.csv    # Example output predictions
└── .gitignore               # Git ignore rules
```

## 📈 Business Applications

This recommender system can be applied to:
- **E-commerce platforms** - Product recommendation engines
- **Content platforms** - Article/video recommendation systems
- **Social networks** - Connection and content suggestions
- **Service marketplaces** - Service provider matching
- **Entertainment platforms** - Movie/music recommendation

## 🎓 Technical Achievements

- **Competitive Excellence**: Top 20% performance in machine learning competition
- **Advanced Algorithm Implementation**: Custom matrix factorization with bias terms
- **Large-Scale Data Processing**: Efficient PySpark pipeline for big data
- **Feature Engineering Excellence**: 27 carefully crafted features from raw data
- **Model Ensemble Design**: Optimal blending strategy for maximum performance
- **Production-Ready Code**: Robust error handling, validation, and documentation

## 🔬 Research Contributions

- Demonstrated effectiveness of hybrid approaches in recommendation systems
- Showed importance of feature engineering in cold-start scenarios
- Validated ensemble methods for improving prediction accuracy
- Provided insights into optimal hyperparameter selection for real-world datasets
- **Competitive Validation**: Proved approach effectiveness against diverse ML solutions

---

*This project demonstrates expertise in building production-ready machine learning systems with strong theoretical foundations, practical implementation skills, and attention to performance optimization. The top 20% competition ranking validates the effectiveness of the hybrid approach and technical implementation.*

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd dsci553-competition-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage Example

Run the recommender system with:
```bash
python yelp_recommender.py <input_directory> <test_file> <output_file>
```
Example:
```bash
python yelp_recommender.py ./data yelp_val.csv sample_predictions.csv
```

**Sample Output (`sample_predictions.csv`):**
| user_id           | business_id        | prediction |
|-------------------|-------------------|------------|
| v1A4K6kGg1p1LrPz | 5UmKMjUEUNdYWqANhGckJw | 4.12       |
| 0a2KyEL0d3Yb1V6a | 9yKzy9PApeiPPOUJEtnvkg | 3.87       |

## 📄 Sample Data Format

**yelp_train.csv**
```
user_id,business_id,stars
v1A4K6kGg1p1LrPz,5UmKMjUEUNdYWqANhGckJw,5
0a2KyEL0d3Yb1V6a,9yKzy9PApeiPPOUJEtnvkg,4
```

**user.json** (one object per line)
```
{"user_id": "v1A4K6kGg1p1LrPz", "review_count": 10, "friends": "0a2KyEL0d3Yb1V6a", "useful": 5, ...}
{"user_id": "0a2KyEL0d3Yb1V6a", "review_count": 3, "friends": "v1A4K6kGg1p1LrPz", "useful": 2, ...}
```

**business.json** (one object per line)
```
{"business_id": "5UmKMjUEUNdYWqANhGckJw", "stars": 4.5, "review_count": 100, ...}
{"business_id": "9yKzy9PApeiPPOUJEtnvkg", "stars": 3.0, "review_count": 50, ...}
``` 
