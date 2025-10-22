# Data Directory

This directory contains the Wine Quality dataset and related data files.

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **Dataset**: Wine Quality (Red Wine)
- **URL**: https://archive.ics.uci.edu/ml/datasets/wine+quality
- **Size**: 1,599 observations, 12 features
- **Target Variable**: Quality (integer rating from 3-8)

## Files

- `winequality-red.csv` - Raw wine quality dataset (downloaded automatically)
- `processed/` - Directory for processed datasets
- `interim/` - Directory for intermediate processing results

## Data Description

### Features (Physicochemical Properties)
1. **fixed acidity** - Tartaric acid concentration (g/L)
2. **volatile acidity** - Acetic acid concentration (g/L)
3. **citric acid** - Citric acid concentration (g/L)
4. **residual sugar** - Sugar remaining after fermentation (g/L)
5. **chlorides** - Salt concentration (g/L)
6. **free sulfur dioxide** - Free SO2 concentration (mg/L)
7. **total sulfur dioxide** - Total SO2 concentration (mg/L)
8. **density** - Wine density (g/cm³)
9. **pH** - Acidity level (0-14 scale)
10. **sulphates** - Potassium sulphate concentration (g/L)
11. **alcohol** - Alcohol percentage (% vol)

### Target Variable
- **quality** - Wine quality rating (integer from 3-8, with 6 being most common)

## Usage Notes

- Dataset will be automatically downloaded when running the analysis
- All preprocessing results will be saved in the `processed/` subdirectory
- Intermediate analysis results stored in `interim/` subdirectory