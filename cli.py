import configargparse

def get_args():

	parser = configargparse.ArgumentParser()

	parser.add_argument(
		'-ts', 
		'--train_size', 
		type=float, 
		default=2/3, 
		help='Proportion of data to allocate to training'
	)

	parser.add_argument(
		'-kf', 
		'--k_folds', 
		type=int, 
		default=5, 
		help='Number of cross-validation folds'
	)

	parser.add_argument(
		'-od', 
		'--output_dim', 
		type=int, 
		choices=[1, 2, 3], 
		help='Dimension of low-dimensional representations'
	)

	parser.add_argument(
		'-cd', 
		'--centre_data', 
		type=bool, 
		default=True, 
		help='If True, low-dimensional representations are centred at the origin'
	)

	parser.add_argument(
		'-r', 
		'--regularization', 
		type=str, 
		choices=['coefficients', 'functional'], 
		help='Type of regularization to apply'
	)

	parser.add_argument(
		'-rr', 
		'--reg_param_range', 
		nargs='+', 
		type=float, 
		help='Lower and upper bounds on the regularization parameter'
	)

	parser.add_argument(
		'-k', 
		'--kernel', 
		type=str, 
		choices=['chi2', 'linear', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'], 
		help='The kernel function'
	)

	parser.add_argument(
		'-dr', 
		'--degree_range', 
		nargs='+', 
		type=int,
		help='Lower and upper bounds on the degree parameter'
	)

	parser.add_argument(
		'-gr', 
		'--gamma_range', 
		nargs='+', 
		type=float,
		help='Lower and upper bounds on the gamma parameter'
	)

	parser.add_argument(
		'-cr', 
		'--coef0_range', 
		nargs='+', 
		type=float,
		help='Lower and upper bounds on the coef0 parameter'
	)

	parser.add_argument(
		'-nt', 
		'--n_trials', 
		type=int, 
		help='Number of hyper-parameter optimization trials'
	)

	return parser.parse_args()