# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg413::cv2.fillPoly+numpy.zeros_like
# name: cv2_numpy_primitive
# summary: Uses cv2.fillPoly, numpy.zeros_like across 2 repos
# anchor_symbols: ['cv2.fillPoly', 'numpy.zeros_like']
# observed in 2 repos: ['jasmcaus__caer', 'ndleah__python-mini-project']...

# --- from ndleah__python-mini-project::Finding_Lanes/lanes.py::roi ---
def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# --- from jasmcaus__caer::caer/transforms/functional.py::_shadow_process ---
def _shadow_process(tens, num_shadows, x1, y1, x2, y2, shadow_dimension) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    tens = _hls(tens) ## Conversion to hls
    cspace = tens.cspace 

    mask = np.zeros_like(tens) 
    imshape = tens.shape

    # Get the list of shadow vertices
    vertices_list = _generate_shadow_coordinates(num_shadows, (x1,y1,x2,y2), shadow_dimension) 

    for vertices in vertices_list: 
        cv.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    # If red channel is hot, the Tensor's "Lightness" channel's brightness value is lowered 
    tens[:,:,1][mask[:,:,0]==255] = tens[:,:,1][mask[:,:,0]==255]*0.5   

    return to_tensor(tens, cspace=cspace)
