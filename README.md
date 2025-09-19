# APR(CS6103) Assignment-1: Stock Price Prediction using Linear Regression

**Author**: Abdul Khazmuddin  
**Roll**: 2511CS06  
**Email**: abdul_2511cs06@iitp.ac.in  
**Phone**: 7003532322  

---

## 📌 Problem Statement
The goal is to build a supervised machine learning model to predict the **next day's closing price** of a stock based on today's market features.

## 📊 Dataset
- Source: Yahoo Finance (`yfinance` library)  
- Stock: Apple Inc. (AAPL)  
- Period: 1 Jan 2020 – 1 Jan 2023  
- Features: Open, High, Low, Volume  
- Target: Next day’s Close price  

## ⚙️ Methodology
1. Download stock data using `yfinance`.  
2. Create target variable (`Close[t+1]`).  
3. Split dataset into train (80%) and test (20%).  
4. Train Linear Regression model.  
5. Evaluate using MSE, MAE, and R² Score.  

## 📈 Results
- **MSE**: 9.31  
- **MAE**: 2.28  
- **R² Score**: 0.99  

Coefficients:  
- Open: -0.89  
- High: +0.90  
- Low: +0.98  
- Volume: ~0  

## ✅ Conclusion
The model shows very high accuracy, but in real-world stock prediction such results are unrealistic due to volatility and external factors. This assignment demonstrates the **application of supervised learning (Linear Regression)** in the financial domain.

---
