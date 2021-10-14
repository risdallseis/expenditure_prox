from numpy.lib.arraysetops import isin
from sklearn.cluster import DBSCAN
import random
from modules.evaluate_model import get_eval_scores
import pandas as pd

def run_dbscan_gs(
    dataframes,
    features,
    epsilon_range,
    min_sample_range,
    iterations,
):
    """This function performs a random grid search for epsilson and min_samples
        this optimisation problem has the bound that only two clusters may form.
    dataframes - LIST 
    features - LIST
    epsilon_range - range of floats
    min_sample_range - range of floats
    """
    results = pd.DataFrame(columns=['iteration','category','precision','recall','f1','eps','m_samples_divisor','features','no_clusters'])
    # FIRST LOOP TO ASSIGN VALUE IN RANGE TO EPS AND MIN_SAMPLES
    for iteration in range(0, iterations):
        epsilon = random.choice(epsilon_range)
        m_samples = random.choice(min_sample_range)
        # SECOND LOOP TO TRY PARAMS FOR EACH DATAFRAME
        for df in dataframes:
            interim_results = {}
            model = DBSCAN(eps=epsilon, min_samples=(len(df))/m_samples)
            df['db_clust'] = pd.Series(model.fit_predict(df[features]), index=df.index)
            no_clusters = len(df['db_clust'].value_counts())
            try:
                pscore, rscore, f1score = get_eval_scores(df['y_true'], df['db_clust'])
                interim_results['iteration'] = iteration
                interim_results['category'] = df.name
                interim_results['precision'] = pscore
                interim_results['recall'] = rscore
                interim_results['f1'] = f1score
                interim_results['eps'] = epsilon
                interim_results['m_samples_divisor'] = m_samples
                interim_results['features'] = [features]
                interim_results['no_clusters'] = no_clusters
                if no_clusters ==2:
                    result_df = pd.DataFrame(data=interim_results)
                    results = pd.concat([results, result_df], axis=0)
            except ValueError:
                pass

    return results
            

def get_best_models(results):
    """Orders models in best performance"""
    best_models = pd.DataFrame(columns=['precision_mean','recall_mean','epsilon','m_samples_divisor'])
    for i in results['iteration'].unique():
        interim_results = {}
        if len(results.loc[results['iteration']==i])==4:
            interim_results['precision_mean'] = results.loc[results['iteration']==i]['precision'].mean()
            interim_results['recall_mean'] = results.loc[results['iteration']==i]['recall'].mean()
            interim_results['f1_mean'] = results.loc[results['iteration']==i]['f1'].mean()
            interim_results['epsilon'] = results.loc[results['iteration']==i]['eps'].values[0]
            interim_results['m_samples_divisor'] = results.loc[results['iteration']==i]['m_samples_divisor'].values[0]
            result_df = pd.DataFrame(data=interim_results, index=[0])
            best_models = pd.concat([best_models, result_df],axis=0)
        else:
            pass
    #best_models['combined_score'] = (best_models['precision_mean']+best_models['recall_mean'])/2
    best_models = best_models.drop_duplicates()
    best_models = best_models.sort_values(by=['f1_mean'], axis=0, ascending=False)

    return best_models


def run_dbscan_gs_noeval(
    dataframes,
    features,
    epsilon_range,
    min_sample_range,
    iterations,
    outlier_prop_target,
):
    """This function performs a random grid search for epsilson and min_samples
        this optimisation problem has the bound that only two clusters may form.
    dataframes - LIST 
    features - LIST
    epsilon_range - range of floats
    min_sample_range - range of floats
    """
    results = pd.DataFrame(columns=['iteration','category','eps','m_samples_divisor',
    'features','no_clusters','outlier_prop','precision','recall','f1'])
    # FIRST LOOP TO ASSIGN VALUE IN RANGE TO EPS AND MIN_SAMPLES
    for iteration in range(0, iterations):
        epsilon = random.choice(epsilon_range)
        m_samples = random.choice(min_sample_range)
        # SECOND LOOP TO TRY PARAMS FOR EACH DATAFRAME
        for df in dataframes:
            interim_results = {}
            model = DBSCAN(eps=epsilon, min_samples=(len(df))/m_samples)
            df['db_clust'] = pd.Series(model.fit_predict(df[features]), index=df.index)
            no_clusters = len(df['db_clust'].value_counts())
            outlier_prop = int((len(df.loc[df['db_clust']==-1])/len(df))*100)
            try:
                interim_results['iteration'] = iteration
                interim_results['category'] = df.name
                interim_results['eps'] = epsilon
                interim_results['m_samples_divisor'] = m_samples
                interim_results['features'] = [features]
                interim_results['no_clusters'] = no_clusters
                interim_results['outlier_prop'] = outlier_prop
                pscore, rscore, f1score = get_eval_scores(df['y_true'], df['db_clust'])
                interim_results['precision'] = pscore
                interim_results['recall'] = rscore
                interim_results['f1'] = f1score

                if outlier_prop in range(outlier_prop_target-5,outlier_prop_target+5,1):
                    if no_clusters ==2:
                        result_df = pd.DataFrame(data=interim_results)
                        results = pd.concat([results, result_df], axis=0)
            except ValueError as err:
                print(err.args)
                pass
    results.drop_duplicates(['category','eps','m_samples_divisor','no_clusters','outlier_prop'],inplace=True)

    return results


def get_best_noevals(results):
    """finds best models for models without evaluation - using mininmum eps param"""
    df = pd.DataFrame(columns=['iteration','category','eps','m_samples_divisor',
    'features','no_clusters','outlier_prop','precision','recall','f1'
    ])
    cats = ['laptops', 'phones', 'tablets','desktops']
    for c in cats:
        df = pd.concat([df, results[results['category']==c].nsmallest(1,'eps')])
    
    return df

