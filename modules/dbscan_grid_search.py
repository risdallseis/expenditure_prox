from sklearn.cluster import DBSCAN
import random
from evaluate_model import get_eval_scores
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
    results = pd.DataFrame(columns=['iteration','category','precision','recall','eps','m_samples_divisor','features','no_clusters'])
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
                pscore, rscore = get_eval_scores(df['y_true'], df['db_clust'])
                interim_results['iteration'] = iteration
                interim_results['category'] = df.name
                interim_results['precision'] = pscore
                interim_results['recall'] = rscore
                interim_results['eps'] = epsilon
                interim_results['m_samples_divisor'] = m_samples
                interim_results['features'] = [features]
                interim_results['no_clusters'] = no_clusters
                result_df = pd.DataFrame(data=interim_results)
                results = pd.concat([results, result_df], axis=0)
            except ValueError:
                pass

    return results
            

    