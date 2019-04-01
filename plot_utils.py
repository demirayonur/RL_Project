import numpy as np
import time
import matplotlib as mpl


def plot_ellipse(ax, pos, cov, color):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 4 * np.sqrt(vals)
    time.sleep(0.1)
    ellip = mpl.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, lw=1, fill=True, alpha=0.2, color=color)

    ax.add_artist(ellip)


def plot_hmm(hmm, axis, coords=(0,1)):
    c1, c2 = coords

    for i in range(hmm.n_state):
        pos = np.array([hmm.means[i,c1], hmm.means[i,c2]])
        cov = hmm.covars[i]
        cov = np.array([[cov[c1,c1], cov[c1,c2]], [cov[c2,c1], cov[c2,c2]]])

        axis.scatter(*pos, color='C{}'.format(i+1), label=str(i+1))
        plot_ellipse(axis, pos, cov, 'C{}'.format(i+1))