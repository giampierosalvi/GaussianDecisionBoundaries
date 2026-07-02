# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gaussian Decision Boundaries
# ## (C) 2017-2020 Giampiero Salvi <giampiero.salvi@ntnu.no>
# This notebook is designed to experiment with the conical decision boundaries that result from a Bayesian classifier with Gaussian class conditional likelihood probability distribution functions.
#
# **NOTE: this is not implemented yet!**
#

# %%
import numpy as np


# %%
def plot_decision_regions(p=0.5, mu1=np.array(), mu2=np.array(), s1=np.array(), s2=np.array(),
                          xlim=np.array(), ylim=np.array(), plotType='Decision Boundary', drawPDFContour=False):
    # greate Multivariate Gaussian objects
    try:
        rv1 = multivariate_normal(mu1, s1)
    except (ValueError, np.linalg.LinAlgError) as e:
        messagebox.showerror("Error!", "Covariance matrix must be positive definite (Gaussian 1)")
        return
    try:
        rv2 = multivariate_normal(mu2, s2)
    except (ValueError, np.linalg.LinAlgError) as e:
        messagebox.showerror("Error!", "Covariance matrix must be positive definite (Gaussian 2)")
    return
    # Compute PDF for a certain range of x and y
    x, y = np.mgrid[xlim[0]:xlim[1]:(xlim[1]-xlim[0])/500.0, ylim[0]:ylim[1]:(ylim[1]-ylim[0])/500.0]
    pos = np.dstack((x, y))
    rv1g = p[0]*rv1.pdf(pos)
    rv2g = p[1]*rv2.pdf(pos)
    sum12 = rv1g+rv2g
    post1 = np.divide(rv1g, sum12)
    post2 = np.divide(rv2g, sum12)
    self.fig.clf()
    #plt.set_cmap('seismic')
    ax = self.fig.add_subplot(111)
    # plot Decision Boundary or Difference of PDFs
    plotType = self.drawType.get()
    if plotType == 'Decision Boundary':
        ax.imshow((post1>post2).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='bwr')
        self.fig.suptitle(plotType)
    elif plotType == 'Log-likelihood ratio':
        maxdata = np.max(np.abs(np.log(rv1.pdf(pos))-np.log(rv2.pdf(pos))))
        cax = ax.imshow((np.log(rv1.pdf(pos)) - np.log(rv2.pdf(pos))).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Spectral_r', vmin=-maxdata, vmax=maxdata)
        self.fig.colorbar(cax)
        self.fig.suptitle('log[p(x|y1)/p(x|y2)]')
    elif plotType == 'Scaled Posterior difference':
        maxdata = np.max(np.abs(rv1g-rv2g))
        cax = ax.imshow((rv1g - rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Spectral_r', vmin=-maxdata, vmax=maxdata)
        self.fig.colorbar(cax)
        self.fig.suptitle('P(y1)p(x|y1) - P(y2)p(x|y2)')
    elif plotType == 'Posterior':
        maxdata = np.max(np.abs(post1))
        cax = ax.imshow((post1).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Spectral_r', vmin=0, vmax=1)
        self.fig.colorbar(cax)
        self.fig.suptitle('P(y1|x) ( = 1 - P(y2|x) )')
    else:
        messagebox.showerror("Error!", "Plot type not supported")
    ax.text(mu1[0], mu1[1], '+', color='white', horizontalalignment='center', verticalalignment='center')
    ax.text(mu2[0], mu2[1], 'o', color='white', horizontalalignment='center', verticalalignment='center')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # plot contours for each PDF
    if drawPDFContour:
        ax.contour(x, y, rv1g, colors='w')
        ax.contour(x, y, rv2g, colors='w')
        #plt.contour(x, y, rv1g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
        #plt.contour(x, y, rv2g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
    self.canvas.draw()


# %%

# %%
