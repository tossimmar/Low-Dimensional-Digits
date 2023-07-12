import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------------------------------

# colour pallette
ocean_breeze = [
	"#007ea7", 
	"#00a8e8", 
	"#91c9e8", 
	"#f0f3bd", 
	"#ffa62b", 
	"#c0d6df", 
	"#7d97ad", 
	"#bae8e8", 
	"#fdffbc", 
	"#ffc977",
]

# --------------------------------------------------------------------------------------------------------------

def add_scatter(fig, x, y=None, z=None, **kwargs): 

	if z is None:
		y = np.zeros(len(x)) if y is None else y
		scatter = go.Scatter(x=x, y=y, **kwargs)
	else:
		scatter = go.Scatter3d(x=x, y=y, z=z, **kwargs)

	fig.add_trace(scatter)

# --------------------------------------------------------------------------------------------------------------

def add_points(fig, labels, x, y=None, z=None, x_neg=None, y_neg=None, z_neg=None):

	add_scatter(
		fig, x, y, z, 
		mode='markers', 
		marker=dict(size=12, color=labels, colorscale=ocean_breeze)
	)

	if x_neg is not None:
		add_scatter(
			fig, x_neg, y_neg, z_neg, 
			mode='markers', 
			marker=dict(size=12, color='white', line=dict(color='black', width=1))
		)

# --------------------------------------------------------------------------------------------------------------

def label_coords(labels, x, y=None, z=None, stat=np.mean):

	uniq_labs = np.unique(labels)
	coords = [[stat(data[l == labels]) for l in uniq_labs] for data in [x, y, z] if data is not None]

	return np.vstack((uniq_labs, coords)).T

# --------------------------------------------------------------------------------------------------------------

def add_labels(fig, labels, x, y=None, z=None, stat=np.mean):

	coords = label_coords(labels, x, y, z, stat)
	d = coords.shape[1] - 1

	add_scatter(
		fig, 
		x=coords[:, 1],
		y=None if d == 1 else coords[:, 2],
		z=None if d in {1, 2} else coords[:, 3],
		mode='text', 
		text=[f'<b>{int(label)}</b>' for label in coords[:, 0]], 
		textposition='bottom center', 
		textfont=dict(color='black', size=16)
	)

# --------------------------------------------------------------------------------------------------------------

def create_figure(title):

	annotations = [
		dict(
			x=0.5, 
			y=0.95, 
			xref='paper', 
			yref='paper', 
			text=title, 
			showarrow=False, 
			font=dict(size=16)
		)
	]

	layout = go.Layout(
		annotations=annotations,
	    margin=dict(t=0, b=0, l=0, r=0), 
	    showlegend=False
	)

	return go.Figure(layout=layout)

# --------------------------------------------------------------------------------------------------------------

def save_html(filename, labels, x, y=None, z=None, x_neg=None, y_neg=None, z_neg=None, title='', stat=np.mean):

	fig = create_figure(title)

	add_points(fig, labels, x, y, z, x_neg, y_neg, z_neg)
	add_labels(fig, labels, x, y, z, stat)

	fig.write_html(f'{filename}.html')