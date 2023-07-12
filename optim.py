from kfda import KFDA

import optuna

from sklearn.model_selection import KFold

# ------------------------------------------------------------------------------------------------------

def create_objective(X, y, k_folds, score_func, output_dim, centre_data, regularization, reg_param_range, kernel, degree_range=None, gamma_range=None, coef0_range=None):
    
    def objective(trial):
        
        reg_param = trial.suggest_float('reg_param', *reg_param_range, log=True)
        
        kernel_args = {}
        if degree_range is not None:
            kernel_args['degree'] = trial.suggest_int('degree', *degree_range)
        if gamma_range is not None:
            kernel_args['gamma'] = trial.suggest_float('gamma', *gamma_range)
        if coef0_range is not None:
            kernel_args['coef0'] = trial.suggest_float('coef0', *coef0_range)
            
        model = KFDA(output_dim, centre_data, regularization, reg_param, kernel, **kernel_args)
        
        avg_score = 0
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
        
        for n, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]
            
            model.fit(X_train, y_train)
            
            y_hat = model.predict(X_valid)
            score = score_func(y_valid, y_hat)
            avg_score = (n * avg_score + score) / (n + 1)
            
        return avg_score
    
    return objective