from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import argparse
from yaml import load, Loader
matplotlib.use('Agg')


def resample(image, pixsize):
    """
    Take a 128 x 128 pixel image, and rebin it such that
    new pixels = pixsize x pixsize old pixels
    """
    assert pixsize in [1, 2, 4, 8, 16] # return error and exit otherwise
    imsize = int(128/pixsize)
    newimage = np.zeros((imsize, imsize))
    for xn, x in enumerate(np.arange(0, 128, pixsize)):
        for yn, y in enumerate(np.arange(0, 128, pixsize)):
            newimage[xn, yn] = np.nansum(image[x:x+pixsize, y:y+pixsize]) # Nansum is important as sum of masked array can be nan
    return newimage


def plot_binned_dph(fig,ax,ax_title,image,pixbin,colormap):
    """
    Plots a dph of a given binning
    
    Inputs:
    fig = figure to which the ax object belongs
    ax = axis object
    ax_title = title required 
    image = the image to be plotted 
    pixbin = the resampling bin size 
    """
 
    im = ax.imshow(resample(image,pixbin),cmap=colormap, interpolation='none')
    ax.set_title(ax_title,fontsize=8)
    ax.set_xlim(-1,128/pixbin - 0.5)
    ax.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
    ax.spines['left'].set_position(('data',-0.5))
    ax.set_yticklabels([])
    ax.xaxis.set_ticks(np.arange(0,128/pixbin,16/pixbin))
    ax.set_xticklabels(np.arange(0,128,16))
    cb = fig.colorbar(im,ax=ax,cmap=colormap, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    return 0

def evt2image(infile, tstart, tend, e_low, e_high):
    hdu = fits.open(infile)
    pixel_edges = np.arange(-0.5, 63.6)

    # e_low = 200 # Energy cut lower bound 
    # e_high = 2000 # Energy cut upper bound

    data = hdu[1].data[np.where( (hdu[1].data['Time'] >= tstart) & (hdu[1].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im1 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im1 = np.transpose(im1[0])
    data = hdu[2].data[np.where( (hdu[2].data['Time'] >= tstart) & (hdu[2].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im2 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im2 = np.transpose(im2[0])
    data = hdu[3].data[np.where( (hdu[3].data['Time'] >= tstart) & (hdu[3].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im3 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im3 = np.transpose(im3[0])
    data = hdu[4].data[np.where( (hdu[4].data['Time'] >= tstart) & (hdu[4].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im4 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im4 = np.transpose(im4[0])

    image = np.zeros((128,128))
    image[0:64,0:64] = im4
    image[0:64,64:128] = im3
    image[64:128,0:64] = im1
    image[64:128,64:128] = im2

    image = np.flip(image,0)

    # plt.imshow(image, origin="lower")
    # plt.show()
    return image

def data_bkgd_image(infile,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend,e_low,e_high):
    """
    Creates source and background dph.
    """
    predph = evt2image(infile,pre_tstart,pre_tend,e_low,e_high)
    grbdph = evt2image(infile,grb_tstart,grb_tend,e_low,e_high)
    postdph = evt2image(infile,post_tstart,post_tend,e_low,e_high)

    bkgddph = predph+postdph

    # oneD_grbdph = grbdph.flatten()
    # oneD_bkgddph = bkgddph.flatten()
    t_src = grb_tend - grb_tstart
    t_total = (pre_tend-pre_tstart)+(post_tend-post_tstart)

    sourcedph = grbdph - bkgddph * t_src/t_total

    return sourcedph


# Make Parser
parser = argparse.ArgumentParser()
parser.add_argument('infile', help='Event file to process', type=str)
parser.add_argument('config', help='Config file containing windows', type=str)
parser.add_argument('--no_plotlc', help='If provided, will not make an lc plot. Only provide if window is set', dest='plotlc', action='store_false')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = load(f, Loader=Loader)


# Window Times
pre_tstart = config['window']['pre_tstart']
pre_tend = config['window']['pre_tend']
grb_tstart = config['window']['grb_tstart']
grb_tend = config['window']['grb_tend']
post_tstart = config['window']['post_tstart']
post_tend = config['window']['post_tend']

# Time to specify as 'middle' of the burst
nominal_grb_time = 0.5*(grb_tstart+grb_tend)

# Get the DPH image
image = data_bkgd_image(args.infile, pre_tstart, pre_tend, grb_tstart,grb_tend, post_tstart, post_tend, config['dph']['emin'], config['dph']['emax'])


#DPH Plotting begins
plotfile = PdfPages(config['outname']['dph'])

fig = plt.figure(figsize=(6,6))
gs = GridSpec(nrows=2, ncols=2)
binnings = [1,4,8,16]

for i in range(4):
    ax = plt.subplot(gs[i])
    ax_title = f"{binnings[i]} Pixel Binning"
    pixbin = binnings[i]
    plot_binned_dph(fig, ax, ax_title, image, pixbin, 'viridis')


plt.text(0.5, 0.95, "Detector Plane Histogram", ha='center', va='top', transform=fig.transFigure)
plt.text(0.5, 0.9, f"at time {nominal_grb_time:2.2f}", ha='center', va='bottom', transform=fig.transFigure)
plotfile.savefig()
plotfile.close()
#DPH Plotting ends


if args.plotlc:
    #LC Plotting begins
    plotfile = PdfPages(config['outname']['lc'])
    plt.figure()
    data = fits.getdata(args.infile, config['lc']['quad'])
    t, e = data['time'], data['energy']

    #Apply energy cuts
    sel = (e>config['dph']['emin']) & (e<config['dph']['emax'])

    tmin = nominal_grb_time - config['lc']['interval']
    tmax = nominal_grb_time + config['lc']['interval']

    #Histogram to get light curve
    counts, bins = np.histogram(t[sel], bins=np.arange(tmin, tmax, config['lc']['tbin']))
    bins = 0.5*(bins[1:] + bins[:-1])

    #Actual Plotting
    plt.plot(bins, counts, 'k')
    plt.axvspan(grb_tstart,grb_tend,color='red',alpha=0.3,label='GRB')
    plt.axvspan(pre_tstart,pre_tend,color='orange',alpha=0.5)
    plt.axvspan(post_tstart,post_tend,color='orange',alpha=0.5,label='Background')
    plt.legend()

    plt.suptitle(f"Light Curve binned at {config['lc']['tbin']} s")
    plt.title("with transient and background windows")
    plt.ylabel("Counts")
    plt.xlabel("Time [s]")
    # plt.show()
    plotfile.savefig()
    plotfile.close()
    #LC plotting ends
