import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yt
import caesar
import seaborn as sns
from tqdm.notebook import tqdm
from scipy.stats import binned_statistic
from scipy.integrate import quad
#import camb
from colossus.cosmology import cosmology
from mycolorpy import colorlist as mcp

colors=mcp.gen_color(cmap="Set1",n=16)
colors2=mcp.gen_color(cmap="Dark2",n=16)

def COlum_H2mass(df_gals,background='default',save=False):
    
    c_alpha = 0.8

    ### CO Lum versus H2 mass
    Bolatto = pd.read_csv('/blue/narayanan/karolina.garcia/github/slick/analysis/Bolattos_data.csv',names=['x','y'])
    #R = pd.Series([float(df_gals['g_Radius'][i].split()[0]) for i in np.arange(len(df_gals['g_Radius']))])
    #M = pd.Series([float(df_gals['g_Mass_H2'][i].split()[0]) for i in np.arange(len(df_gals['g_Mass_H2']))])
    R = df_gals['g_Radius']
    M = df_gals['g_Mass_H2']
    if background=='dark_background':
        sns.set(font_scale = 2,rc={'figure.figsize':(13.7,11.27)},style="ticks", context="talk")#,palette='Spectral')
        plt.style.use(background)
        ax3 = sns.scatterplot(data = 10**Bolatto, x='x',y='y',s=200,alpha=0.8,label='Observations (Bolatto et al., 2013)',color='orange')
        sns.set(palette='Spectral')
        ax3 = sns.scatterplot(x=(df_gals['CO10_intTB'])*(np.pi/2)*(R)**2, y=M, s=200,alpha=0.8,label='Simulations (Garcia et al., in prep)')
        ax3.set(xscale='log',yscale='log')
        ax3.set(xlabel='CO Luminosity [K km s$^{-1}$ pc$^2$]',ylabel='H$_2$ Mass [$M_{\odot}$]')
        ax3.set(xlim=(0,1e12))
        legend = plt.legend(loc=4,prop={'size':20},fontsize='x-large')
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_frame().set_facecolor('black')
        #legend.get_frame().set
        if save:
            plt.savefig('figures/COlum_vs_H2mass_'+background+'.png')
        else: plt.show()
    else:
        fig = plt.figure()
        plt.style.use('seaborn-v0_8-deep')
        ax3= fig.add_subplot(111)
        #ax3.scatter(data = 10**Bolatto, x='x',y='y',s=20,color=colors[2],alpha=c_alpha,edgecolor='white',linewidth=0.5,label='Observed MW clusters') #color='blue'
        #ax3.scatter(x=(df_gals['CO10_intTB'])*(np.pi/2)*(R)**2, y=M, s=20,color=colors2[2],alpha=c_alpha,edgecolor='white',linewidth=0.5,label='SLICK galaxies z=0') #color='darkorange'
        ax3.scatter(x=(df_gals['CO10_intTB']), y=M, s=20,color=colors2[2],alpha=c_alpha,edgecolor='white',linewidth=0.5,label='SLICK galaxies z=0') #color='darkorange'
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        #ax3.set_xlim(0,1e12)
        ax3.set_xlabel('CO Luminosity [K km s$^{-1}$ pc$^2$]',fontsize=13)
        ax3.set_ylabel('H$_2$ Mass [$M_{\odot}$]',fontsize=13)
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        legend = plt.legend(loc='lower right',prop={'size':11})
        legend.get_frame().set_facecolor('white')
        if save:
            plt.savefig('figures/COlum_vs_H2mass_'+background+'.pdf')
        else: plt.show()