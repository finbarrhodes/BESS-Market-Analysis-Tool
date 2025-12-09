# GB BESS Market Analysis & Forecasting Tool 🔋

The goals for this project is the following: A comprehensive Python toolkit for analyzing Great Britain's Battery Energy Storage System (BESS) markets, including historical trend analysis, price forecasting, and market insights.

## 📋 Project Overview

This project provides:
- **Data Collection**: Automated scraping from National Grid ESO and Elexon BMRS APIs
- **Market Analysis**: Trend analysis across DC, DR, DM, BM, and FR markets
- **Price Forecasting**: Multiple ML and statistical models for price prediction
- **Visualization**: Interactive Plotly dashboards and Streamlit web app
- **Reporting**: Automated report generation with key insights

## 🎯 Key Features

- Historical data collection from public APIs (2020-present)
- Supply-demand driver analysis
- Regulatory change impact assessment
- Multiple forecasting models (ARIMA, Prophet, scikit-learn)
- Model comparison and evaluation
- Interactive web dashboard
- Automated report generation

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. **Navigate to the project directory**:
```bash
cd gb-bess-market-analysis
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up API credentials**:
   - Register for Elexon BMRS API: https://www.elexonportal.co.uk/
   - Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
   - Add your API key to `.env`:
```
ELEXON_API_KEY=your_actual_api_key_here
```

5. **Configure the project**:
   - Edit `config.yaml` to adjust date ranges, markets, and analysis parameters

## 📁 Project Structure

```
gb-bess-market-analysis/
├── data/
│   ├── raw/              # Raw data from APIs
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data_collection/  # API scrapers and data ingestion
│   │   ├── neso_collector.py      # National Grid ESO collector
│   │   ├── elexon_collector.py    # Elexon BMRS collector
│   │   └── collect_data.py        # Main collection script
│   ├── analysis/         # Market trend analysis
│   ├── forecasting/      # Prediction models
│   ├── visualization/    # Plotting and dashboard code
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Generated reports and figures
├── tests/                # Unit tests
├── logs/                 # Application logs
├── config.yaml           # Project configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Usage

### 1. Data Collection

Collect historical market data:

```bash
# Collect data from both sources for a date range
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31

# Collect only from National Grid ESO
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31 --sources neso

# Collect only from Elexon BMRS
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31 --sources elexon

# Use date range from config.yaml
python src/data_collection/collect_data.py
```

**Note**: The collectors are set up with proper structure but will need the actual API endpoints verified from the data portals, as these can change. Check:
- National Grid ESO: https://data.nationalgrideso.com/
- Elexon BMRS: https://www.elexonportal.co.uk/

### 2. Run Analysis

Analyze market trends (coming soon):
```bash
python src/analysis/market_analysis.py
```

### 3. Generate Forecasts

Train and evaluate forecasting models (coming soon):
```bash
python src/forecasting/train_models.py
```

### 4. Launch Dashboard

Start the interactive Streamlit dashboard (coming soon):
```bash
streamlit run src/visualization/dashboard.py
```

## 📊 Key Markets Analyzed

- **Dynamic Containment (DC)**: Fast-acting frequency response
- **Dynamic Regulation (DR)**: Continuous frequency regulation
- **Dynamic Moderation (DM)**: Slower frequency response
- **Balancing Mechanism (BM)**: Energy balancing market
- **Frequency Response (FR)**: Traditional frequency services
- **Day-Ahead Market**: Energy arbitrage opportunities

## 🔮 Forecasting Models

The project will implement and compare multiple forecasting approaches:

1. **Statistical Models**:
   - ARIMA (AutoRegressive Integrated Moving Average)
   - SARIMA (Seasonal ARIMA)
   - Prophet (Facebook's time series forecasting)

2. **Machine Learning Models**:
   - Random Forest
   - Gradient Boosting
   - Feature engineering with lag and rolling statistics

3. **Model Evaluation**:
   - RMSE, MAE, MAPE metrics
   - Cross-validation
   - Residual analysis

## 📈 Data Sources

- **National Grid ESO**: System frequency, demand data
  - Website: https://www.nationalgrideso.com/
  - Data Portal: https://data.nationalgrideso.com/
  
- **Elexon BMRS**: Market prices, imbalance data
  - Website: https://www.elexon.co.uk/
  - Portal: https://www.elexonportal.co.uk/
  - API Docs: https://www.elexon.co.uk/documents/training-guidance/bsc-guidance-notes/bmrs-api-and-data-push-user-guide/

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📝 Development Roadmap

- [x] Project setup and structure
- [x] Configuration and utilities
- [x] API data collection modules
- [ ] Data cleaning and preprocessing
- [ ] Exploratory data analysis notebook
- [ ] Market trend analysis
- [ ] Forecasting model implementation
- [ ] Model comparison and evaluation
- [ ] Interactive dashboard
- [ ] Automated report generation
- [ ] Documentation and examples

## ⚠️ Important Notes

### API Endpoints
The data collectors are structured with proper error handling and rate limiting, but you'll need to:
1. Verify the actual API endpoints from the official documentation
2. Check the response structure and adjust parsing accordingly
3. Some endpoints may require authentication beyond just an API key

### Data Availability
- Dynamic Containment data is typically available from 2020 onwards
- Some historical data may have different formats
- Check data availability windows for each market

### Rate Limits
- National Grid ESO: Currently set to 100 requests/minute
- Elexon BMRS: Currently set to 60 requests/minute
- Adjust in config.yaml if needed

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

## 📄 License

This project is for educational and portfolio purposes.

## 🙏 Acknowledgments

- National Grid ESO for providing open data access
- Elexon for BMRS data and documentation
- UK battery storage community for insights

## 📧 Contact

[Your Name] - [Your Email/LinkedIn]

---

**Note**: This project is for analysis and educational purposes only. Not financial or trading advice.
