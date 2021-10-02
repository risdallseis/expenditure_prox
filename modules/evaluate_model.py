from sklearn.metrics import precision_recall_curve, precision_score, recall_score

def get_eval_scores(
    y_true,
    y_pred,
    print_results = False,
):
    """Calculates precision and recall scores of input y-vals"""
    pscore = precision_score(y_true, y_pred, pos_label=-1)
    rscore = recall_score(y_true, y_pred, pos_label=-1)
    if print_results:
        print(f"The precision score is: {pscore} and the recall is {rscore}")

    return pscore, rscore

