import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import sobel, gaussian_filter
from constants import MAX_RADIUS, MIN_RADIUS, RADIUS_STEP, ANNULUS_WIDTH, EDGE_THRESHOLD, NEG_INTERIOR_WEIGHT

def _detectEdges(image, threshold):
    image = sobel(image, 0)**2 + sobel(image, 1)**2 
    image -= image.min()
    
    image = image > image.max()*threshold
    image.dtype = np.int8
        
    return image

def _makeAnnulusKernel(outer_radius, annulus_width):
    grids = np.mgrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]

    kernel_template = grids[0]**2 + grids[1]**2
    
    outer_circle = kernel_template <= outer_radius**2
    inner_circle = kernel_template < (outer_radius - annulus_width)**2
    
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle*NEG_INTERIOR_WEIGHT
    annulus = outer_circle - inner_circle
    return annulus

def _detectCircles(image, radii, annulus_width):
    acc = np.zeros((radii.size, image.shape[0], image.shape[1]))

    for i, r in enumerate(radii):
        C = _makeAnnulusKernel(r, annulus_width)
        acc[i,:,:] = fftconvolve(image, C, 'same')

    return acc

def _displayResults(image, edges, center, radius, output=None):
    plt.gray()
    fig = plt.figure(1)
    fig.clf()
    subplots = []
    subplots.append(fig.add_subplot(1, 2, 1))
    plt.imshow(edges)
    plt.title('Edge image')
    
    subplots.append(fig.add_subplot(1, 2, 2))
    plt.imshow(image)
    plt.title('Center: %s, Radius: %d' % (str(center), radius))
    
    blob_circ = plt_patches.Circle(center,radius, fill=False, ec='red')
    plt.gca().add_patch(blob_circ)
    
    plt.axis('image')

    if output:
        plt.savefig(output)
        
    plt.draw()
    plt.show()
    
    return

def _topNCircles(acc, radii, n):
    maxima = []
    max_positions = []
    max_signal = 0
    for i, r in enumerate(radii):
        max_positions.append(np.unravel_index(acc[i].argmax(), acc[i].shape))
        maxima.append(acc[i].max())

        signal = maxima[i]/np.sqrt(float(r))
        
        if signal > max_signal:
            max_signal = signal
            (circle_y, circle_x) = max_positions[i]
            radius = r
        
    return (circle_x, circle_y), radius

def DetectCircleFromFile(filename, show_result=False):
    image = plt.imread(filename)
    center, radius = DetectCircle(image, True, show_result)
    return center, radius

def DetectCircle(image, preprocess=False, show_result=False):
    if preprocess:
        if image.ndim > 2:
            image = np.mean(image, axis=2)
        
        image = gaussian_filter(image, 2)
        
        edges = _detectEdges(image, EDGE_THRESHOLD)
        edge_list = np.array(edges.nonzero())
        density = float(edge_list[0].size)/edges.size
            
    radii = np.arange(MIN_RADIUS, MAX_RADIUS, RADIUS_STEP)
    acc = _detectCircles(edges, radii, ANNULUS_WIDTH)
    center, radius = _topNCircles(acc, radii, 1)
    
    if show_result:
        _displayResults(image, edges, center, radius)
                        
    return center, radius

def test():
    DetectCircleFromFile('images\christmasBall.jpg', True)
        
if __name__ == "__main__":
    test()