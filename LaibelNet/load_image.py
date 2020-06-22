from PIL import Image


def load_image(image_path, image_size):
    '''

    Args:
        image_path:
        image_size:

    Returns:

    '''
    im = Image.open(image_path)
    w, h = im.size
    res = min(h, w)
    h0, w0 = (h - res) // 2, (w - res) // 2
    im = im.resize(image_size, box=(w0, h0, w0 + res, h0 + res))
    im = im.convert('RGB')
    return im
