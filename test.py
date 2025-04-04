from inference_pipeline import preprocess_and_predict

sample_input = {
    'age': 32,
    'annual_income': 85000,
    'net_worth': 90000,
    'debt_to_income_ratio': 0.25,
    'savings_rate': 0.12,
    'portfolio_value': 60000,
    'financial_knowledge_score': 7,
    'macroeconomic_score': 6,
    'sentiment_index': 5,
    'equity_allocation_pct': 60,
    'fixed_income_allocation_pct': 30,
    'monthly_contribution': 1500,
    'market_volatility_index': 22,
    'employment_status': 'Employed',
    'risk_appetite': 'Medium',
    'investment_horizon_years': 10,
    'month': '2023-06'
}

predicted = preprocess_and_predict(sample_input)
print("📈 Predicted Strategy:", predicted)
