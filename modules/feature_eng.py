"""Module for engineering features, to be done after preproc
"""
from nltk.tree import Tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def no_negative_reviews(list_scores):
    """Gets number of negative reviews"""
    no_reviews = 0
    if isinstance(list_scores, list):   
        for item in list_scores:
            item = int(item)
            if item < 3:
                no_reviews += 1

    return no_reviews


def no_positive_reviews(list_scores):
    """Gets number of positive reviews"""
    no_reviews = 0
    if isinstance(list_scores, list):   
        for item in list_scores:
            item = int(item)
            if item > 3:
                no_reviews += 1

    return no_reviews 


def get_y_true(df):
	"""Get true y values based on total_sales zscore"""
	df['zscore'] = (df['TOTAL_SALES']-df['TOTAL_SALES'].mean())/df['TOTAL_SALES'].std()
	df.loc[(df['zscore'] >=1), 'y_true'] = -1
	df.loc[(df['zscore'] <1), 'y_true'] = 0

	return df.drop('zscore', axis=1)


def generate_features(df):
    """Generates some features after preproc"""
    df['neg_reviews'] = df['review_rating'].apply(lambda x: no_negative_reviews(x))
    df['pos_reviews'] = df['review_rating'].apply(lambda x: no_positive_reviews(x))
    df['price/Rvol'] = df['price']/df['no_reviews']
    df['Rvol/%rec'] = df['no_reviews']/df['recommendation_percent']
    df['posR/Rvol'] = df['pos_reviews']/df['no_reviews']
    df['negR/Rvol'] = df['neg_reviews']/df['no_reviews']
    
    return df


def do_PCA(
	df,
	keep_no_reviews=True,
	return_original_df=True,
	no_components = 2,
):
	"""Performs principal component analysis on data. Fills NA with 0.
	"""
	pca = PCA(no_components)
	df1 = df[['pos_reviews','neg_reviews', 'price/Rvol', 'Rvol/%rec', 'posR/Rvol', 
			'negR/Rvol','price','no_reviews','recommendation_percent',
			'summary_star_rating','TOTAL_SALES','product_name'
			]]
	if keep_no_reviews == False:
		df1 = df1.loc[df1.no_reviews !=0]
	sales = df1['TOTAL_SALES']
	names = df1['product_name']
	df1 = df1.drop(['TOTAL_SALES','product_name'], axis=1)
	df1 = (df1-df1.mean())/df1.std()
	df1.fillna(0, inplace=True)
	pC = pca.fit_transform(df1)
	df_pc = pd.DataFrame(data = pC, columns = ['c1','c2'])
	if return_original_df == False:
		df_pc.reset_index(inplace=True)
		sales.reset_index(inplace=True)
		names.reset_index(inplace=True)
		pc_df = pd.concat([df_pc, sales,names], axis=1, ignore_index=False)
	else:
		df_pc.reset_index( inplace=True)
		df.reset_index(inplace=True)
		pc_df = pd.concat([df, df_pc],axis=1, ignore_index=False)


	return pc_df