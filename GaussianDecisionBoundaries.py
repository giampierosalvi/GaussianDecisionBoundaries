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
import matplotlib.backends.backend_tkagg as tkagg

def about():
   aboutText = "GaussianDecisionBoundaries.py\n\n(C) 2017 Giampiero Salvi\n\nDraws the decision boundary between two Gaussian distributions according to the\nMaximum a Posteriori criterium. You can change the a priori probabilities, the\nmean vectors and covariance matrices. You can also show the difference between\nthe two Probability Densidty Functions (PDFs) and display contours of the original PDFs."
   messagebox.showinfo("About", aboutText)

def redraw(fig):
   # acquie Gaussian parameters
   p = np.array([float(p1.get()), 1.0-float(p1.get())])
   mu1 = np.array([float(mu1x.get()), float(mu1y.get())])
   mu2 = np.array([float(mu2x.get()), float(mu2y.get())])
   s1 = np.array([[float(s1x.get()), float(s1xy.get())],
                  [float(s1xy.get()), float(s1y.get())]])
   s2 = np.array([[float(s2x.get()), float(s2xy.get())],
                  [float(s2xy.get()), float(s2y.get())]])
   # greate Multivariate Gaussian objects
   try:
      rv1 = multivariate_normal(mu1, s1)
   except ValueError:
      messagebox.showerror("Error!", "Covariance matrix must be positive semidefinite (Gaussian 1)")
   try:
      rv2 = multivariate_normal(mu2, s2)
   except ValueError:
      messagebox.showerror("Error!", "Covariance matrix must be positive semidefinite (Gaussian 2)")
   # Compute PDF for a certain range of x and y
   xlim = [float(xmin.get()), float(xmax.get())]
   ylim = [float(ymin.get()), float(ymax.get())]
   x, y = np.mgrid[xlim[0]:xlim[1]:(xlim[1]-xlim[0])/500.0, ylim[0]:ylim[1]:(ylim[1]-ylim[0])/500.0]
   pos = np.dstack((x, y))
   rv1g = p[0]*rv1.pdf(pos)
   rv2g = p[1]*rv2.pdf(pos)
   fig.clf()
   ax = fig.add_subplot(111)
   # plot Decision Boundary or Difference of PDFs
   if drawType.get() == 'Decision Boundary':
      ax.imshow((rv1g>rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
   else:
      cax = ax.imshow((rv1g-rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
      fig.colorbar(cax)
   ax.text(mu1[0], mu1[1], '+', color='white', horizontalalignment='center', verticalalignment='center')
   ax.text(mu2[0], mu2[1], 'o', color='white', horizontalalignment='center', verticalalignment='center')
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   fig.suptitle('Gaussian Decision Boundaries')
   # plot contours for each PDF
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
xmin = tk.StringVar()
xmax = tk.StringVar()
ymin = tk.StringVar()
ymax = tk.StringVar()

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
xmin.set("-1.5")
xmax.set("1.5")
ymin.set("-1.5")
ymax.set("1.5")

# create control widgets
entryWidth=5
controlFrame = ttk.Frame(root)
gaussianFrame = ttk.Frame(controlFrame)
gaussian1Frame = ttk.LabelFrame(gaussianFrame, text='Gaussian 1')
p1W = ttk.Entry(gaussian1Frame, textvariable=p1, width=entryWidth)
mu1xW = ttk.Entry(gaussian1Frame, textvariable=mu1x, width=entryWidth)
mu1yW = ttk.Entry(gaussian1Frame, textvariable=mu1y, width=entryWidth)
s1xW = ttk.Entry(gaussian1Frame, textvariable=s1x, width=entryWidth)
s1yW = ttk.Entry(gaussian1Frame, textvariable=s1y, width=entryWidth)
s1xyW = ttk.Entry(gaussian1Frame, textvariable=s1xy, width=entryWidth)
s1yxW = ttk.Entry(gaussian1Frame, textvariable=s1xy, width=entryWidth)
gaussian2Frame = ttk.LabelFrame(gaussianFrame, text='Gaussian 2')
p2W = ttk.Label(gaussian2Frame, text='1-p1')
mu2xW = ttk.Entry(gaussian2Frame, textvariable=mu2x, width=entryWidth)
mu2yW = ttk.Entry(gaussian2Frame, textvariable=mu2y, width=entryWidth)
s2xW = ttk.Entry(gaussian2Frame, textvariable=s2x, width=entryWidth)
s2yW = ttk.Entry(gaussian2Frame, textvariable=s2y, width=entryWidth)
s2xyW = ttk.Entry(gaussian2Frame, textvariable=s2xy, width=entryWidth)
s2yxW = ttk.Entry(gaussian2Frame, textvariable=s2xy, width=entryWidth)
drawW = ttk.LabelFrame(controlFrame, text='Drawing')
drawTypeW = ttk.Combobox(drawW, textvariable=drawType)
drawTypeW['values'] = ('Decision Boundary', 'PDF Difference')
drawPDFContourW = ttk.Checkbutton(drawW, text="Draw PDF Contours", variable=drawPDFContour)
xlimFrame = ttk.Frame(drawW)
xlimL = ttk.Label(xlimFrame, text='xlim')
xminW = ttk.Entry(xlimFrame, textvariable=xmin, width=entryWidth)
xmaxW = ttk.Entry(xlimFrame, textvariable=xmax, width=entryWidth)
ylimFrame = ttk.Frame(drawW)
ylimL = ttk.Label(ylimFrame, text='ylim')
yminW = ttk.Entry(ylimFrame, textvariable=ymin, width=entryWidth)
ymaxW = ttk.Entry(ylimFrame, textvariable=ymax, width=entryWidth)
redrawButton = ttk.Button(drawW, text="Redraw", command=lambda: redraw(fig))
aboutButton = ttk.Button(drawW, text="About...", command=about)

# place widgets within gaussian1Frame
gaussian1Frame.grid(column=0, row=0, columnspan=2, rowspan=6)
p1L = ttk.Label(gaussian1Frame, text='p1')
p1L.grid(row=0, column=0)
p1W.grid(row=0, column=1)
mu1L = ttk.Label(gaussian1Frame, text='mean1')
mu1L.grid(row=1, column=0, columnspan=2)
mu1xW.grid(row=2, column=0)
mu1yW.grid(row=2, column=1)
s1L = ttk.Label(gaussian1Frame, text='cov1')
s1L.grid(row=3, column=0, columnspan=2)
s1xW.grid(row=4, column=0)
s1xyW.grid(row=4, column=1)
s1yxW.grid(row=5, column=0)
s1yW.grid(row=5, column=1)

# place widgets within gaussian2Frame
gaussian2Frame.grid(column=0, row=0, columnspan=2, rowspan=6)
p2L = ttk.Label(gaussian2Frame, text='p2 = 1-p1')
p2L.grid(row=0, column=0, columnspan=2)
mu2L = ttk.Label(gaussian2Frame, text='mean2')
mu2L.grid(row=1, column=0, columnspan=2)
mu2xW.grid(row=2, column=0)
mu2yW.grid(row=2, column=1)
s2L = ttk.Label(gaussian2Frame, text='cov2')
s2L.grid(row=3, column=0, columnspan=2)
s2xW.grid(row=4, column=0)
s2xyW.grid(row=4, column=1)
s2yxW.grid(row=5, column=0)
s2yW.grid(row=5, column=1)

# place widgets within drawing frame
drawTypeW.pack(side="top")
drawPDFContourW.pack(side="top")
#xlimFrame.grid(column=0, row=0, columnspan=3, rowspan=1)
#xlimL.grid(row=0, column=0)
#xminW.grid(row=0, column=1)
#xmaxW.grid(row=0, column=2)
xlimL.pack(side="left")
xminW.pack(side="left")
xmaxW.pack(side="left")
xlimFrame.pack(side="top")
#ylimFrame.grid(column=0, row=0, columnspan=3, rowspan=1)
#ylimL.grid(row=0, column=0)
#yminW.grid(row=0, column=1)
#ymaxW.grid(row=0, column=2)
ylimL.pack(side="left")
yminW.pack(side="left")
ymaxW.pack(side="left")
ylimFrame.pack(side="top")
redrawButton.pack(side="top")
aboutButton.pack(side="top")

# place Gaussian frames within gaussianFrame
gaussian1Frame.pack(side="left")
gaussian2Frame.pack(side="left")

# place gfame and drawing frame within controlFrame
gaussianFrame.pack(side="top")
drawW.pack(side="top")

controlFrame.pack(side="left")

frame = tk.Frame(root)
fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, master=root)
tkagg.NavigationToolbar2TkAgg(canvas, root)
canvas.show()
canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
frame.pack()
redraw(fig)
root.mainloop()
