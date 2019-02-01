#!/usr/bin/env python
# GaussianDecisionBoundaries.py
#
# Draws the decision boundary between two Gaussian distributions according to
# the Maximum a Posteriori criterium. You can change the a priori probabilities,
# the mean vectors and covariance matrices. You can also show the difference
# between the two Probability Densidty Functions (PDFs) and display contours
# of the original PDFs.
#
# Copyright (C) 2017-2018 Giampiero Salvi <giampi@kth.se>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LogNorm
from tkinter import messagebox
#import matplotlib.backends.backend_tkagg as tkagg

class GaussianDecisionBoundaries(ttk.Frame):
   def __init__(self, parent):
      ttk.Frame.__init__(self, parent)
      self.parent = parent
      # Gaussian distribution parameters
      self.p1 = tk.StringVar()
      self.mu1x = tk.StringVar()
      self.mu1y = tk.StringVar()
      self.s1x = tk.StringVar()
      self.s1y = tk.StringVar()
      self.s1xy = tk.StringVar()
      self.p2 = tk.StringVar()
      self.mu2x = tk.StringVar()
      self.mu2y = tk.StringVar()
      self.s2x = tk.StringVar()
      self.s2y = tk.StringVar()
      self.s2xy = tk.StringVar()
      # drawing parameters
      self.drawType = tk.StringVar()
      self.drawPDFContour = tk.BooleanVar()
      self.xmin = tk.StringVar()
      self.xmax = tk.StringVar()
      self.ymin = tk.StringVar()
      self.ymax = tk.StringVar()
      # initialize parameters
      self.set_defaults()
      # initialize user interface
      self.initUI()

   # set default values
   def set_defaults(self):
      self.p1.set('0.5')
      self.mu1x.set('-1.0')
      self.mu1y.set('-1.0')
      self.s1x.set('1.0')
      self.s1y.set('1.0')
      self.s1xy.set('0.0')
      self.p2.set('0.5')
      self.mu2x.set('1.0')
      self.mu2y.set('1.0')
      self.s2x.set('1.0')
      self.s2y.set('1.0')
      self.s2xy.set('0.0')
      self.drawType.set('Decision Boundary')
      self.drawPDFContour.set(True)
      self.xmin.set("-1.5")
      self.xmax.set("1.5")
      self.ymin.set("-1.5")
      self.ymax.set("1.5")

   def initUI(self):
      # set default font size
      defaultFont = font.nametofont("TkDefaultFont")
      defaultFont.configure(family="helvetica", size=14)
      # create control widgets
      entryWidth=5
      controlFrame = ttk.Frame(self.parent)
      gaussianParameterFrame = ttk.Frame(controlFrame)
      gaussian1ParameterFrame = ttk.LabelFrame(gaussianParameterFrame, text='Gaussian 1')
      e = dict()
      e['p1W'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.p1, width=entryWidth, font=defaultFont)
      e['mu1xW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.mu1x, width=entryWidth, font=defaultFont)
      e['mu1yW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.mu1y, width=entryWidth, font=defaultFont)
      e['s1xW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.s1x, width=entryWidth, font=defaultFont)
      e['s1yW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.s1y, width=entryWidth, font=defaultFont)
      e['s1xyW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.s1xy, width=entryWidth, font=defaultFont)
      e['s1yxW'] = ttk.Entry(gaussian1ParameterFrame, textvariable=self.s1xy, width=entryWidth, font=defaultFont)
      gaussian2ParameterFrame = ttk.LabelFrame(gaussianParameterFrame, text='Gaussian 2')
      e['p2W'] = ttk.Label(gaussian2ParameterFrame, text='1-p1')
      e['mu2xW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.mu2x, width=entryWidth, font=defaultFont)
      e['mu2yW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.mu2y, width=entryWidth, font=defaultFont)
      e['s2xW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.s2x, width=entryWidth, font=defaultFont)
      e['s2yW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.s2y, width=entryWidth, font=defaultFont)
      e['s2xyW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.s2xy, width=entryWidth, font=defaultFont)
      e['s2yxW'] = ttk.Entry(gaussian2ParameterFrame, textvariable=self.s2xy, width=entryWidth, font=defaultFont)
      drawingControlFrame = ttk.LabelFrame(controlFrame, text='Drawing')
      drawTypeW = ttk.Combobox(drawingControlFrame, textvariable=self.drawType, font=defaultFont, postcommand=self.redraw)
      drawTypeW['values'] = ('Decision Boundary', 'Posterior', 'Scaled Posterior difference', 'Log-likelihood ratio')
      drawPDFContourW = ttk.Checkbutton(drawingControlFrame, text="Draw PDF Contours", variable=self.drawPDFContour, command=self.redraw)
      xlimFrame = ttk.Frame(drawingControlFrame)
      xlimL = ttk.Label(xlimFrame, text='xlim')
      e['xminW'] = ttk.Entry(xlimFrame, textvariable=self.xmin, width=entryWidth, font=defaultFont)
      e['xmaxW'] = ttk.Entry(xlimFrame, textvariable=self.xmax, width=entryWidth, font=defaultFont)
      ylimFrame = ttk.Frame(drawingControlFrame)
      ylimL = ttk.Label(ylimFrame, text='ylim')
      e['yminW'] = ttk.Entry(ylimFrame, textvariable=self.ymin, width=entryWidth, font=defaultFont)
      e['ymaxW'] = ttk.Entry(ylimFrame, textvariable=self.ymax, width=entryWidth, font=defaultFont)
      redrawButton = ttk.Button(drawingControlFrame, text="Redraw", command=self.redraw)
      aboutButton = ttk.Button(controlFrame, text="About...", command=self.about)
      resetButton = ttk.Button(controlFrame, text="Reset", command=self.set_defaults)
      # bindings
      for key in e.keys():
         e[key].bind('<Return>', self.redraw)
      drawTypeW.bind("<<ComboboxSelected>>", self.redraw)
      # place widgets within gaussian1ParameterFrame
      p1L = ttk.Label(gaussian1ParameterFrame, text='p1')
      p1L.grid(row=0, column=0)
      e['p1W'].grid(row=0, column=1)
      mu1L = ttk.Label(gaussian1ParameterFrame, text='mean1')
      mu1L.grid(row=1, column=0, columnspan=2)
      e['mu1xW'].grid(row=2, column=0)
      e['mu1yW'].grid(row=2, column=1)
      s1L = ttk.Label(gaussian1ParameterFrame, text='cov1')
      s1L.grid(row=3, column=0, columnspan=2)
      e['s1xW'].grid(row=4, column=0)
      e['s1xyW'].grid(row=4, column=1)
      e['s1yxW'].grid(row=5, column=0)
      e['s1yW'].grid(row=5, column=1)
      # place widgets within gaussian2ParameterFrame
      p2L = ttk.Label(gaussian2ParameterFrame, text='p2 = 1-p1')
      p2L.grid(row=0, column=0, columnspan=2)
      mu2L = ttk.Label(gaussian2ParameterFrame, text='mean2')
      mu2L.grid(row=1, column=0, columnspan=2)
      e['mu2xW'].grid(row=2, column=0)
      e['mu2yW'].grid(row=2, column=1)
      s2L = ttk.Label(gaussian2ParameterFrame, text='cov2')
      s2L.grid(row=3, column=0, columnspan=2)
      e['s2xW'].grid(row=4, column=0)
      e['s2xyW'].grid(row=4, column=1)
      e['s2yxW'].grid(row=5, column=0)
      e['s2yW'].grid(row=5, column=1)
      # place widgets within drawing frame
      drawTypeW.pack(side="top")
      drawPDFContourW.pack(side="top")
      xlimL.pack(side="left")
      e['xminW'].pack(side="left")
      e['xmaxW'].pack(side="left")
      xlimFrame.pack(side="top")
      ylimL.pack(side="left")
      e['yminW'].pack(side="left")
      e['ymaxW'].pack(side="left")
      ylimFrame.pack(side="top")
      redrawButton.pack(side="top")
      # place Gaussian frames within gaussianParameterFrame
      gaussian1ParameterFrame.pack(side="left")
      gaussian2ParameterFrame.pack(side="left")
      # place gfame and drawing frame within controlFrame
      aboutButton.pack(side="top")
      resetButton.pack(side="top")
      gaussianParameterFrame.pack(side="top")
      drawingControlFrame.pack(side="top")
      controlFrame.pack(side="left")
      # define drawing frame
      drawingFrame = tk.Frame(self.parent)
      self.fig = plt.Figure()
      self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
      #tkagg.NavigationToolbar2TkAgg(canvas, root)
      self.canvas.draw()
      self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
      drawingFrame.pack()
      self.redraw()
      
   def about(self):
      aboutText = "GaussianDecisionBoundaries.py\n\n(C) 2017-2018 Giampiero Salvi\n\nDraws the decision boundary between two Gaussian distributions according to the Maximum a Posteriori criterion. You can change the a priori probabilities, the mean vectors and covariance matrices. You can also show the difference between the two Probability Densidty Functions (PDFs) and display contours of the original PDFs.\n\nSource code at: https://github.com/giampierosalvi/GaussianDecisionBoundaries"
      messagebox.showinfo("About", aboutText)
      
   def redraw(self, event=None):
      # acquie Gaussian parameters
      p = np.array([float(self.p1.get()), 1.0-float(self.p1.get())])
      mu1 = np.array([float(self.mu1x.get()), float(self.mu1y.get())])
      mu2 = np.array([float(self.mu2x.get()), float(self.mu2y.get())])
      s1 = np.array([[float(self.s1x.get()), float(self.s1xy.get())],
                     [float(self.s1xy.get()), float(self.s1y.get())]])
      s2 = np.array([[float(self.s2x.get()), float(self.s2xy.get())],
                     [float(self.s2xy.get()), float(self.s2y.get())]])
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
      xlim = [float(self.xmin.get()), float(self.xmax.get())]
      ylim = [float(self.ymin.get()), float(self.ymax.get())]
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
         self.fig.suptitle('log[p(x|1)/p(x|2)]')
      elif plotType == 'Scaled Posterior difference':
         maxdata = np.max(np.abs(rv1g-rv2g))
         cax = ax.imshow((rv1g - rv2g).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Spectral_r', vmin=-maxdata, vmax=maxdata)
         self.fig.colorbar(cax)
         self.fig.suptitle('P(1)p(x|1) - P(2)p(x|2)')
      elif plotType == 'Posterior':
         maxdata = np.max(np.abs(post1))
         cax = ax.imshow((post1).T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='Spectral_r', vmin=0, vmax=1)
         self.fig.colorbar(cax)
         self.fig.suptitle('P(1|x) ( = 1 - P(2|x) )')
      else:
         messagebox.showerror("Error!", "Plot type not supported")
      ax.text(mu1[0], mu1[1], '+', color='white', horizontalalignment='center', verticalalignment='center')
      ax.text(mu2[0], mu2[1], 'o', color='white', horizontalalignment='center', verticalalignment='center')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      # plot contours for each PDF
      if self.drawPDFContour.get():
         ax.contour(x, y, rv1g, colors='w')
         ax.contour(x, y, rv2g, colors='w')
         #plt.contour(x, y, rv1g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
         #plt.contour(x, y, rv2g.reshape(x.shape), norm=LogNorm(vmin=1.0, vmax=40.0),levels=np.logspace(0, 3, 10))
      self.canvas.draw()

def main():
   root = tk.Tk()
   root.title("Gaussian Decision Boundaries")
   gdbs = GaussianDecisionBoundaries(root)
   root.mainloop()

if __name__ == '__main__':
   main()
