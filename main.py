import cli
import kfda 
import optim 
import scatter

import optuna
import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------------------

def main():

	# Get command line arguments
	# --------------------------
	args = cli.get_args()

	# Load digits data
	# ----------------
	digits = load_digits()
	X, y = digits['data'], digits['target']
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=args.train_size)

	# Hyper-parameter optimization
	# ----------------------------
	objective = optim.create_objective(
	    X=X_train, 
	    y=y_train, 
	    k_folds=args.k_folds, 
	    score_func=accuracy_score, 
	    output_dim=args.output_dim,
	    centre_data=args.centre_data,
	    regularization=args.regularization,
	    reg_param_range=args.reg_param_range,
	    kernel=args.kernel,
	    degree_range=args.degree_range,
	    gamma_range=args.gamma_range, 
	    coef0_range=args.coef0_range
	)
	study = optuna.create_study(study_name='KFDA Study', direction='maximize')
	study.optimize(objective, n_trials=args.n_trials)
	best_params = study.best_trial.params

	# Fit optimal model
	# -----------------
	kernel_args = {k: v for k, v in best_params.items() if k != 'reg_param'}
	model = kfda.KFDA(
		output_dim=args.output_dim, 
		centre_data=args.centre_data, 
		regularization=args.regularization, 
		reg_param=best_params['reg_param'], 
		kernel=args.kernel, 
		**kernel_args
	)
	model.fit(X_train, y_train)

	# Plot low-dimensional training representations
	# ---------------------------------------------
	train_projections = model.transform(X_train)
	scatter.save_html(
		filename='kfda-train-plot', 
		labels=y_train, 
		x=train_projections if model.output_dim == 1      else train_projections[:, 0], 
		y=None              if model.output_dim == 1      else train_projections[:, 1], 
		z=None              if model.output_dim in {1, 2} else train_projections[:, 2], 
		title=f'{model.output_dim}-Dimensional Training Representations', 
		stat=np.median
	)

	# Plot low-dimensional testing representations
	# --------------------------------------------
	valid_projections = model.transform(X_valid)
	predictions = model.predict(X_valid)

	pos_labels = y_valid[y_valid == predictions]
	pos_projections = valid_projections[y_valid == predictions]

	neg_projections = valid_projections[y_valid != predictions]
	
	scatter.save_html(
		filename='kfda-test-plot', 
		labels=pos_labels, 
		x=pos_projections     if model.output_dim == 1      else pos_projections[:, 0], 
		y=None                if model.output_dim == 1      else pos_projections[:, 1], 
		z=None                if model.output_dim in {1, 2} else pos_projections[:, 2],
		x_neg=neg_projections if model.output_dim == 1      else neg_projections[:, 0],
		y_neg=None            if model.output_dim == 1      else neg_projections[:, 1],
		z_neg=None            if model.output_dim in {1, 2} else neg_projections[:, 2],
		title=f'{model.output_dim}-Dimensional Testing Representations', 
		stat=np.median
	)

# ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
