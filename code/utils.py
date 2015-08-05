import numpy as np
from skimage import transform

class FlatMapper(object):
    """
    Render a flat field patch from a larger flat field, and vice versa.
    """
    def __init__(self, flat_field, patch_side, pixel_locations):
        self.create_lookup(flat_field, patch_side, pixel_locations)
        self.flat = flat_field

    def create_lookup(self, flat_field, patch_side, pixel_locations):
        """
        Create index to lookup patches of flat field.
        """
        rows = pixel_locations[0]
        cols = pixel_locations[1]
        
        buff = (patch_side - 1) / 2
        c, r = np.meshgrid(range(-buff, buff + 1),
                           range(-buff, buff + 1))
        self.rowinds = (r[None, :, :] + rows[:, None, None]).astype(np.int)
        self.colinds = (c[None, :, :] + cols[:, None, None]).astype(np.int)

    def get_2D_flat_patches(self):
        """
        Return the flat patches for a given patch.
        """
        return self.flat[self.rowinds, self.colinds]

    def get_1D_flat_patches(self):
        """
        Return flattened flat patches.
        """
        patches = self.get_2D_flat_patches()
        return patches.reshape(patches.shape[0], patches.shape[1] ** 2)

def tweak_psf(psf_model, shear, scale):
    """
    Apply an affine transformation to the model.
    """
    shift_y, shift_x = np.array(psf_model.shape[:2]) / 2.
    tf_shift = transform.AffineTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.AffineTransform(translation=[shift_x, shift_y])
    tf = transform.AffineTransform(scale=scale, shear=shear)
    return transform.warp(psf_model, (tf_shift + (tf + tf_shift_inv)).inverse)

if __name__ == '__main__':

    rows = np.array([3])
    cols = np.array([4])
    flat = np.arange(72).reshape(8, 9)

    f = FlatMapper(flat, 5, (rows, cols))
    print flat
    print f.get_flat()
