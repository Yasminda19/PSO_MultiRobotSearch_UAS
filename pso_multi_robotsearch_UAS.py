import numpy as np
import pyswarms as ps
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
from pyswarms.utils.plotters.formatters import Mesher, Designer

import random

def signal_intensity(x):
    
    i = 2.0/x ** 2 +  1

    return i

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':1.2, 'k': 2, 'p': 2}

# Call instance of PSO
optimizer = ps.single.LocalBestPSO(n_particles=20, dimensions=2, options=options)

# Perform optimization
best_cost, best_pos = optimizer.optimize(signal_intensity, iters=1000)

#plot cost history
plot_cost_history(optimizer.cost_history)
plt.show()

m = Mesher(func=fx.sphere)

# Make animation
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0.0,0.0))

# Simpan gif dalam jupyternotebook
animation.save('plot0.gif', writer='imagemagick', fps=10)
Image(url='plot0.gif')
