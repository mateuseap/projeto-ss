import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import sobel, gaussian_filter
from constants import RADIUS_STEP, ANNULUS_WIDTH, EDGE_THRESHOLD, NEG_INTERIOR_WEIGHT

MAX_RADIUS = 150
MIN_RADIUS = 75

title_placeholder = st.empty()

max_radius_slider_value = st.slider("Selecione o valor do **RAIO MÁXIMO** da bola a ser detectada:", 10, 1000, MAX_RADIUS)
min_radius_slider_value = st.slider("Selecione o valor do **RAIO MÍNIMO** da bola a ser detectada:", 1, 700, MIN_RADIUS)

MAX_RADIUS = max_radius_slider_value
MIN_RADIUS = min_radius_slider_value

def detect_edges(image, threshold):
    image = sobel(image, 0)**2 + sobel(image, 1)**2 
    image -= image.min()
    
    image = image > image.max()*threshold
    image.dtype = np.int8
        
    return image

def make_annulus_kernel(outer_radius, annulus_width):
    grids = np.mgrid[-outer_radius:outer_radius+1, -outer_radius:outer_radius+1]

    kernel_template = grids[0]**2 + grids[1]**2
    
    outer_circle = kernel_template <= outer_radius**2
    inner_circle = kernel_template < (outer_radius - annulus_width)**2
    
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle*NEG_INTERIOR_WEIGHT
    annulus = outer_circle - inner_circle
    return annulus

def detect_circles(image, radii, annulus_width):
    acc = np.zeros((radii.size, image.shape[0], image.shape[1]))

    for i, r in enumerate(radii):
        C = make_annulus_kernel(r, annulus_width)
        acc[i,:,:] = fftconvolve(image, C, 'same')

    return acc

def display_results(image, edges, center, radius):
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
    
    return fig

def top_n_circles(acc, radii, n):
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

def detect_circle_from_file(image):
    output_img = detect_circle(image, True)
    return output_img

def detect_circle(image, preprocess=False):
    if preprocess:
        if image.ndim > 2:
            image = np.mean(image, axis=2)
        
        image = gaussian_filter(image, 2)
        
        edges = detect_edges(image, EDGE_THRESHOLD)
        edge_list = np.array(edges.nonzero())
        density = float(edge_list[0].size)/edges.size
            
    radii = np.arange(MIN_RADIUS, MAX_RADIUS, RADIUS_STEP)
    acc = detect_circles(edges, radii, ANNULUS_WIDTH)
    center, radius = top_n_circles(acc, radii, 1)

    output_image = display_results(image, edges, center, radius)
    return output_image
        
def app():
    title_placeholder.title("Detector de bolas natalinas")

    uploaded_image = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

    image_placeholder = st.empty()
    upload_success_message_placeholder = st.empty()
    output_image_placeholder = st.empty()
    detect_success_message_placeholder = st.empty()

    detect_button = st.button("Detectar")

    if uploaded_image is not None:
        upload_success_message_placeholder.success("Imagem carregada com sucesso!")
        image_placeholder.image(uploaded_image)

    if detect_button and uploaded_image is not None:
        upload_success_message_placeholder.empty()
        image = plt.imread(uploaded_image)
        output_image = detect_circle_from_file(image)

        if output_image is not None:
            output_image_placeholder.pyplot(output_image)
            detect_success_message_placeholder.success("Detecção concluída!")

app()