# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg669::cv2.calcHist+cv2.imread+cv2.split
# name: cv2_numpy_primitive
# summary: Uses cv2.calcHist, cv2.imread, cv2.split, numpy.argmax across 2 repos
# anchor_symbols: ['cv2.calcHist', 'cv2.imread', 'cv2.split', 'numpy.argmax']
# observed in 2 repos: ['ahmetozlu__tensorflow_object_counting_api', 'ahmetozlu__vehicle_counting_tensorflow']...

# --- from ahmetozlu__tensorflow_object_counting_api::utils/color_recognition_module/color_histogram_feature_extraction.py::color_histogram_of_test_image ---
def color_histogram_of_test_image(test_src_image):
	#load the image
	image = test_src_image

	chans = cv2.split(image)
	colors = ("b", "g", "r")
	features = []
	feature_data = ""
	counter = 0
	for (chan, color) in zip(chans, colors):
		counter = counter + 1
	
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
	
		# find the peak pixel values for R, G, and B
		elem = np.argmax(hist)

		if (counter == 1):
			blue = str(elem)
		elif (counter ==2):
			green = str(elem)
		elif (counter ==3):
			red = str(elem)
			feature_data = red + "," + green + "," + blue
	with open(current_path+"/utils/color_recognition_module/"+"test.data", "w") as myfile:						
		myfile.write(feature_data)

# --- from ahmetozlu__vehicle_counting_tensorflow::utils/color_recognition_module/color_histogram_feature_extraction.py::color_histogram_of_test_image ---
def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
    with open(current_path + '/utils/color_recognition_module/'
              + 'test.data', 'w') as myfile:
        myfile.write(feature_data)

# --- from ahmetozlu__tensorflow_object_counting_api::utils/color_recognition_module/color_histogram_feature_extraction.py::color_histogram_of_training_image ---
def color_histogram_of_training_image(img_name):
	
	# detect image color by using image file name to label training data
	if "red" in img_name:
		data_source = "red"
	elif "yellow" in img_name:
		data_source = "yellow"
	elif "green" in img_name:
		data_source = "green"
	elif "orange" in img_name:
		data_source = "orange"
	elif "white" in img_name:
		data_source = "white"
	elif "black" in img_name:
		data_source = "black"
	elif "blue" in img_name:
		data_source = "blue"
	elif "violet" in img_name:
		data_source = "violet"

	#load the image
	image = cv2.imread(img_name)

	chans = cv2.split(image)
	colors = ("b", "g", "r")
	features = []
	feature_data = ""
	counter = 0
	for (chan, color) in zip(chans, colors):
		counter = counter + 1
		
		hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		
		# find the peak pixel values for R, G, and B
		elem = np.argmax(hist)

		if (counter == 1):
			blue = str(elem)
		elif (counter ==2):
			green = str(elem)
		elif (counter ==3):
			red = str(elem)
			feature_data = red + "," + green + "," + blue

	with open("training.data", "a") as myfile:		
		myfile.write(feature_data + "," + data_source + "\n")

# --- from ahmetozlu__vehicle_counting_tensorflow::utils/color_recognition_module/color_histogram_feature_extraction.py::color_histogram_of_training_image ---
def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')
