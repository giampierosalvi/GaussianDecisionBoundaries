# GaussianDecisionBoundaries.py
#
# Draws the decision boundary between two Gaussian distributions according to the
# Maximum a Posteriori criterium. You can change the a priori probabilities, the
# mean vectors and covariance matrices. You can also show the difference between
# the two Probability Densidty Functions (PDFs) and display contours of the original
# PDFs.
#
# TODO
# - adjust axis limits depending on the Gaussian parameters
# - 
#
# (C) 2017 Giampiero Salvi <giampi@kth.se>
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm
from tkinter import messagebox

def redraw(fig):
   p = np.array([float(p1.get()), 1.0-float(p1.get())])
   mu1 = np.array([float(mu1x.get()), float(mu1y.get())])
   mu2 = np.array([float(mu2x.get()), float(mu2y.get())])
   s1 = np.array([[float(s1x.get()), float(s1xy.get())],
                  [float(s1xy.get()), float(s1y.get())]])
   s2 = np.array([[float(s2x.get()), float(s2xy.get())],
                  [float(s2xy.get()), float(s2y.get())]])
   try:
      rv1 = multivariate_normal(mu1, s1)
   except ValueError:
      messagebox.showerror("Error!", "Covariance matrix must be positive semidefinite (Gaussian 1)")
   try:
      rv2 = multivariate_normal(mu2, s2)
   except ValueError:
      messagebox.showerror("Error!", "Covariance matrix must be positive semidefinite (Gaussian 2)")
   xlim = [-1.5, 1.5]
   ylim = [-1.5, 1.5]
   x, y = np.mgrid[xlim[0]:xlim[1]:0.003, ylim[0]:ylim[1]:0.003]
   pos = np.dstack((x, y))
   rv1g = p[0]*rv1.pdf(pos)
   rv2g = p[1]*rv2.pdf(pos)
   fig.clf()
   ax = fig.add_subplot(111)
   if drawType.get() == 'Decision Boundary':
      ax.imshow((rv1g>rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
   else:
      ax.imshow((rv1g-rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
   ax.text(mu1[0], mu1[1], '+', color='white', horizontalalignment='center', verticalalignment='center')
   ax.text(mu2[0], mu2[1], 'o', color='white', horizontalalignment='center', verticalalignment='center')
   fig.suptitle('Gaussian Decision Boundaries')
   if drawPDFContour.get():
      ax.contour(x, y, rv1g, colors='w')
      ax.contour(x, y, rv2g, colors='w')
      #plt.contour(x, y, rv1g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
      #plt.contour(x, y, rv2g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
   canvas.draw()

root = tk.Tk()
root.title("Gaussian Decision Boundaries")

# decides the type of plot
#plotTypeVar = tk.StringVar()
#plotTypeVar.set('Decision Boundary')
#plotTypeW = ttk.Combobox(root, textvariable=plotTypeVar)
#plotTypeW['values'] = ('Decision Boundary', 'Contours', 'Gaussian 1 colormap', 'Gaussian 2 colormap')
#plotTypeW.bind('<<ComboboxSelected>>', redraw)
#plotTypeW.pack()

# Gaussian distribution parameters
p1 = tk.StringVar()
mu1x = tk.StringVar()
mu1y = tk.StringVar()
s1x = tk.StringVar()
s1y = tk.StringVar()
s1xy = tk.StringVar()
p2 = tk.StringVar()
mu2x = tk.StringVar()
mu2y = tk.StringVar()
s2x = tk.StringVar()
s2y = tk.StringVar()
s2xy = tk.StringVar()

# drawing parameters
drawType = tk.StringVar()
drawPDFContour = tk.BooleanVar()

# set default values
p1.set('0.5')
mu1x.set('-1.0')
mu1y.set('-1.0')
s1x.set('1.0')
s1y.set('1.0')
s1xy.set('0.0')
p2.set('0.5')
mu2x.set('1.0')
mu2y.set('1.0')
s2x.set('1.0')
s2y.set('1.0')
s2xy.set('0.0')
drawType.set('Decision Boundary')
drawPDFContour.set(True)

# create control widgets
entryWidth=5
controlFrame = ttk.Frame(root)
gframe = ttk.Frame(controlFrame)
g1W = ttk.LabelFrame(gframe, text='Gaussian 1')
p1W = ttk.Entry(g1W, textvariable=p1, width=entryWidth)
mu1xW = ttk.Entry(g1W, textvariable=mu1x, width=entryWidth)
mu1yW = ttk.Entry(g1W, textvariable=mu1y, width=entryWidth)
s1xW = ttk.Entry(g1W, textvariable=s1x, width=entryWidth)
s1yW = ttk.Entry(g1W, textvariable=s1y, width=entryWidth)
s1xyW = ttk.Entry(g1W, textvariable=s1xy, width=entryWidth)
s1yxW = ttk.Entry(g1W, textvariable=s1xy, width=entryWidth)
g2W = ttk.LabelFrame(gframe, text='Gaussian 2')
p2W = ttk.Label(g2W, text='1-p1')
mu2xW = ttk.Entry(g2W, textvariable=mu2x, width=entryWidth)
mu2yW = ttk.Entry(g2W, textvariable=mu2y, width=entryWidth)
s2xW = ttk.Entry(g2W, textvariable=s2x, width=entryWidth)
s2yW = ttk.Entry(g2W, textvariable=s2y, width=entryWidth)
s2xyW = ttk.Entry(g2W, textvariable=s2xy, width=entryWidth)
s2yxW = ttk.Entry(g2W, textvariable=s2xy, width=entryWidth)
drawW = ttk.LabelFrame(controlFrame, text='Drawing')
drawTypeW = ttk.Combobox(drawW, textvariable=drawType)
drawTypeW['values'] = ('Decision Boundary', 'PDF Difference')
drawPDFContourW = ttk.Checkbutton(drawW, text="Draw PDF Contours", variable=drawPDFContour)
button = ttk.Button(drawW, text="Redraw", command=lambda: redraw(fig))

# place widgets within Gaussian 1 frame
g1W.grid(column=0, row=0, columnspan=2, rowspan=6)
p1L = ttk.Label(g1W, text='p1')
p1L.grid(row=0, column=0)
p1W.grid(row=0, column=1)
mu1L = ttk.Label(g1W, text='mean1')
mu1L.grid(row=1, column=0, columnspan=2)
mu1xW.grid(row=2, column=0)
mu1yW.grid(row=2, column=1)
s1L = ttk.Label(g1W, text='cov1')
s1L.grid(row=3, column=0, columnspan=2)
s1xW.grid(row=4, column=0)
s1xyW.grid(row=4, column=1)
s1yxW.grid(row=5, column=0)
s1yW.grid(row=5, column=1)

# place widgets within Gaussian 2 frame
g2W.grid(column=0, row=0, columnspan=2, rowspan=6)
p2L = ttk.Label(g2W, text='p2 = 1-p1')
p2L.grid(row=0, column=0, columnspan=2)
mu2L = ttk.Label(g2W, text='mean2')
mu2L.grid(row=1, column=0, columnspan=2)
mu2xW.grid(row=2, column=0)
mu2yW.grid(row=2, column=1)
s2L = ttk.Label(g2W, text='cov2')
s2L.grid(row=3, column=0, columnspan=2)
s2xW.grid(row=4, column=0)
s2xyW.grid(row=4, column=1)
s2yxW.grid(row=5, column=0)
s2yW.grid(row=5, column=1)

# place widgets within drawing frame
drawTypeW.pack(side="top")
drawPDFContourW.pack(side="top")
button.pack(side="top")

# place Gaussian frames within gframe
g1W.pack(side="left")
g2W.pack(side="left")

# place gfame and drawing frame within controlFrame
gframe.pack(side="top")
drawW.pack(side="top")

controlFrame.pack(side="left")

frame = tk.Frame(root)
fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
frame.pack()
redraw(fig)
root.mainloop()
