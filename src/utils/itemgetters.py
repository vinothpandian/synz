
from operator import itemgetter

get_size = itemgetter("width", "height")
get_rect_data = itemgetter("xmin", "ymin", "width", "height")
get_image_size = itemgetter("image_width", "image_height")
