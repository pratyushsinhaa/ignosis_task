import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
from scipy import stats
import json

def load_and_examine_data():
    purchase_df = pd.read_csv('purchase_behaviour.csv')
    transaction_df = pd.read_csv('transaction_data.csv')
    
    print("\nDataset Info:")
    print("\nPurchase Behavior Data:")
    print(purchase_df.info())
    print("\nTransaction Data:")
    print(transaction_df.info())
    
    return purchase_df, transaction_df

def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna({
        col: df[col].mode()[0] if df[col].dtype == 'object' 
        else df[col].mean() for col in df.columns
    })
    return df

def analyze_customers(merged_df):
    customer_frequency = merged_df.groupby('LYLTY_CARD_NBR').size()
    loyal_threshold = np.percentile(customer_frequency, 80)
    loyal_customers = customer_frequency[customer_frequency >= loyal_threshold].index
    
    loyal_profile = merged_df[merged_df['LYLTY_CARD_NBR'].isin(loyal_customers)]
    
    return loyal_profile

def analyze_products(merged_df):
    product_metrics = merged_df.groupby('PROD_NBR').agg({
        'TOT_SALES': ['count', 'sum'],
        'LYLTY_CARD_NBR': 'nunique'
    }).round(2)
    
    product_metrics.columns = ['sales_count', 'total_revenue', 'unique_customers']
    product_metrics['avg_revenue_per_customer'] = (
        product_metrics['total_revenue'] / product_metrics['unique_customers']
    ).round(2)
    
    return product_metrics.nlargest(3, 'total_revenue')

def create_visualizations(loyal_profile, top_products):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(data=loyal_profile, y='LIFESTAGE', ax=axes[0,0])
    axes[0,0].set_title('Lifestage Distribution of Loyal Customers')
    
    sns.countplot(data=loyal_profile, x='PREMIUM_CUSTOMER', ax=axes[0,1])
    axes[0,1].set_title('Premium vs Regular Customers')
    
    sns.barplot(data=top_products.reset_index(), 
                x='PROD_NBR', y='total_revenue', ax=axes[1,0])
    axes[1,0].set_title('Top 3 Products by Revenue')
    
    sns.boxplot(data=loyal_profile, y='TOT_SALES', ax=axes[1,1])
    axes[1,1].set_title('Sales Distribution')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')

    try:
        purchase_df, transaction_df = load_and_examine_data()
        purchase_df = clean_data(purchase_df)
        transaction_df = clean_data(transaction_df)
        
        merged_df = pd.merge(transaction_df, purchase_df, 
                           on='LYLTY_CARD_NBR',
                           how='left')
        
        loyal_profile = analyze_customers(merged_df)
        top_products = analyze_products(merged_df)
        
        create_visualizations(loyal_profile, top_products)
        
        summary = {
            "top_products": top_products.to_dict(),
            "loyal_customer_profile": {
                "lifestage_distribution": loyal_profile['LIFESTAGE'].value_counts().to_dict(),
                "premium_customer_ratio": loyal_profile['PREMIUM_CUSTOMER'].value_counts().to_dict(),
                "avg_transaction_value": loyal_profile['TOT_SALES'].mean().round(2)
            },
            "hypothesis": "Loyal customers are likely to be older families and young families who prefer budget and mainstream products. They might be attracted to these products due to their affordability and value for money, which is crucial for families managing expenses."
        }
        
        with open('analysis_results.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        print("\nAnalysis complete! Check analysis_results.png and analysis_results.json")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
