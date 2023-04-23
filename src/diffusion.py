import numpy as np
import time

class Diffusion:
    cumulative = None
    l = None

    def __init__(self, l, size, t_max):
        # when close to zero, only a tight range of coefficients are modified at each time step
        range_coefficients = 0.1
        # what kind of noise patterns can we generate? exponent of frequency decay
        decay_exponent = -1.0
        # large if images generated this way have large absolute intensities
        noise_magnitude = 0.01 

        i = np.reshape(np.linspace(-0.5, 0.5, size), (size, -1))  
        t = np.reshape(np.linspace(-0.5, 0.5, t_max), (-1, t_max))

        sigma = noise_magnitude*np.exp(-(i+t)**2/range_coefficients)
        decay = np.reshape(np.arange(1,size+1)**decay_exponent, (size, -1))

        sigma = sigma

        self.cumulative = np.cumsum(sigma, axis=1)
        self.sigma = sigma
        self.l = l
        self.t_max = t_max
        self.size = size

    def sample(self, vec, t):
        noise = np.random.default_rng(seed=int(time.time())).normal(loc=0, scale=self.cumulative[:,t])

        fv = np.sqrt((self.t_max-t)/self.t_max)
        fn = np.sqrt(1-fv)

        return fv * vec + fn * noise
            
        

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import itertools

if __name__ == '__main__':
    d = Diffusion(0.5, 100, 100)


    plt.matshow(d.sigma)

#    fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
#
#    X, Y = np.array(list(itertools.product(range(100),range(100)))).T
#    Z = d.cumulative[X, Y]
#
#    axs.scatter(X, Y, Z)
#
#    # Set the orthographic projection.
#    axs.set_proj_type('ortho')  # FOV = 0 deg
#    axs.set_title("'ortho'\nfocal_length = ∞", fontsize=10)
    plt.show()

#    print(d.cumulative)

    
