# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg36::matplotlib.backends.backend_tkagg.FigureCanvasTkAgg+matplotlib.figure.Figure
# name: matplotlib_primitive
# summary: Uses matplotlib.backends.backend_tkagg.FigureCanvasTkAgg, matplotlib.figure.Figure across 3 repos
# anchor_symbols: ['matplotlib.backends.backend_tkagg.FigureCanvasTkAgg', 'matplotlib.figure.Figure']
# observed in 3 repos: ['donnemartin__data-science-ipython-notebooks', 'jasmcaus__caer', 'ndleah__python-mini-project']...

# --- from donnemartin__data-science-ipython-notebooks::scikit-learn/fig_code/svm_gui.py::View.__init__ ---
def __init__(self, root, controller):
        f = Figure()
        ax = f.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas.mpl_connect('key_press_event', self.onkeypress)
        canvas.mpl_connect('key_release_event', self.onkeyrelease)
        canvas.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        self.shift_down = False
        self.controllbar = ControllBar(root, controller)
        self.f = f
        self.ax = ax
        self.canvas = canvas
        self.controller = controller
        self.contours = []
        self.c_labels = None
        self.plot_kernels()

# --- from ndleah__python-mini-project::Finance_Tracker/main.py::Application.__init__ ---
def __init__(self):
        super().__init__()

        # configure window
        self.title("Finance Tracker")
        self.geometry(f"{800}x{500}")

        # configure grid layout
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=5)

        self.income = 0
        self.expense = 0
        self.income_transactions = [] 
        self.expense_transactions = []

        # create main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Finance Tracker", 
                                                 font=ctk.CTkFont(size=30, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Adding navigation buttons
        self.income_button = ctk.CTkButton(self.sidebar_frame, text="Income", command=self.income_button_event)
        self.income_button.grid(row=1, column=0, padx=20, pady=20)
        self.expenses_button = ctk.CTkButton(self.sidebar_frame, text="Expenses", command=self.expenses_button_event)
        self.expenses_button.grid(row=2, column=0, padx=20, pady=20)
        self.balance_button = ctk.CTkButton(self.sidebar_frame, text="Balance", command=self.balance_button_event)
        self.balance_button.grid(row=3, column=0, padx=20, pady=20)

        # Create Entry fields(User can type the text)
        self.income_frame, self.income_tree = self.create_transaction_frame("Income", self.add_income, row=1, transactions=self.income_transactions)
        self.expense_frame, self.expense_tree = self.create_transaction_frame("Expense", self.add_expense, row=2, transactions=self.expense_transactions)
        self.balance_frame = self.create_balance_frame(row=3)

        self.hide_frames()
        self.income_frame.grid()

        # Plot
        self.fig = Figure(figsize = (4, 4), dpi = 100) 
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.balance_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0)

# --- from jasmcaus__caer::examples/GUI/caer_gui.py::main ---
def main():
    global root
    global canvas
    global fig
    global subplot
    global lblError
    global lblFileName
    global currentImage
    global previous_image
    global transformedImage
    global imageSelection
    global popup_menu_image
    global reload_local_file
    global showAxis
    global resizedImgBtn
    global flip_H
    global flip_V
    global btnFlip_H
    global btnFlip_V
    global rotateImgBtn
    global selectedSize
    global selectedAngle
    global rotationAngle
    global anchor
    global anchorSelection
    global rotationApplied
    global lblCurrentAngle
    global currentAngle
    global imgGamma
    global hue
    global saturation
    global sharpen
    global gaussian_blur
    global posterize
    global solarize
    global show_edges
    global low_threshold
    global sliderLowThreshold
    global show_emboss
    global emboss
    global embossKernel
    global sliderEmboss
    global sobel_threshold

    # create our main window
    root = Tk()
    root.config(background='white')
    root.title('CAER Image GUI - Python v' + pythonVersion)
    root.geometry('1024x768')
    root.bind('<Destroy>', on_exit)

    # the following works for a single screen setup
    # if using a multi-screen setup then see the following link:
    # Ref: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/56913005#56913005
    screenDPI = root.winfo_fpixels('1i')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    currentImage = None
    previous_image = 'Island'
    transformedImage = None
    reload_local_file = False
    rotationApplied = False
    anchor = None
    showAxis = False
    flip_H, flip_V = False, False
    currentAngle = 0.0

    embossKernel = caer.data.np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    #-----------------------------------------------------------------------

    # add a frame to hold top controls
    frame1 = Frame(root, background='black')
    frame1.pack(side=TOP, fill=X)

    # create the built-in image selection variable and choices
    imageSelection = StringVar()
    imageChoices = ['Open File >>', 'Bear', 'Beverages', 'Black Cat', 'Camera', 'Gold Fish', 'Guitar', 'Island', 'Mountain', 'Night', 'Puppies', 'Snow', 'Sunrise', 'Sea Turtle', 'Tent']
    imageSelection.set('Island')
    imageSelection.trace('w', show_original_image)

    # create the built-in image selection popup menu
    popup_menu_image = OptionMenu(frame1, imageSelection, *imageChoices)
    popup_menu_image['width'] = 10
    popup_menu_image['bg'] = 'lightgreen'
    popup_menu_image.pack(side=LEFT, padx=2)

    # create a button to re-size the image
    resizedImgBtn = Button(frame1, text='Resize', width=6, bg='lightgrey', relief=RAISED, command=resize_image)
    resizedImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create an entry box for re-size dimensions
    selectedSize = StringVar()
    resizedImgSize = Entry(frame1, justify=CENTER, textvariable=selectedSize, font='Helvetica 10', width=10, bg='white', relief=RAISED)
    resizedImgSize.pack(side=LEFT, padx=2, pady=2)
    selectedSize.set('')

    # create a button to rotate the image
    rotateImgBtn = Button(frame1, text='Rotate', width=6, bg='lightgrey', relief=RAISED, command=show_rotated_image)
    rotateImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation angle
    lblAngle = Label(frame1, text=u'\u2220', fg='yellow', bg='black', font='Helvetica 10')
    lblAngle.pack(side=LEFT, padx=2, pady=2)

    # create the rotation angle selection variable and an entry box
    selectedAngle = StringVar()
    rotationAngle = Entry(frame1, justify=CENTER, textvariable=selectedAngle, font='Helvetica 10', width=5, bg='white', relief=RAISED)
    rotationAngle.pack(side=LEFT, padx=2, pady=2)
    selectedAngle.set('0.0')

    # create a read-only label for the current angle
    lblCurrentAngle = Label(frame1, text='0.0', state='disabled', fg='lightgrey', bg='white', font='Helvetica 8', width=5)
    lblCurrentAngle.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation anchor
    lblAnchor = Label(frame1, text=u'\u2693', fg='yellow', bg='black', font='Helvetica 10')
    lblAnchor.pack(side=LEFT, padx=2, pady=2)

    # create the rotation anchor selection variable and choices
    anchorSelection = StringVar()
    anchorChoices = ['BottomLeft', 'BottomMiddle', 'BottomRight', 'Center', 'MiddleLeft', 'MiddleRight', 'TopLeft', 'TopMiddle', 'TopRight']
    anchorSelection.set('Center')

    # create the anchor selection popup menu
    popup_menu_anchor = OptionMenu(frame1, anchorSelection, *anchorChoices)
    popup_menu_anchor['width'] = 12
    popup_menu_anchor.pack(side=LEFT, padx=2)

    # create a label to show the name of the local image file opened by user
    lblFileName = Label(frame1, text='', fg='yellow', bg='black', font='Helvetica 10')
    lblFileName.pack(side=RIGHT, padx=10, pady=2)

    #-----------------------------------------------------------------------

    # add a frame to hold side controls, screen attributes and the Error labels
    frame2 = Frame(root, background='black')
    frame2.pack(side=RIGHT, fill=Y)

    # create the image gamma slider control
    imgGamma = DoubleVar()
    sliderGamma = Scale(frame2, label='Gamma', variable=imgGamma, troughcolor='blue', from_=0.1, to=2.0, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGamma.pack(side=TOP, anchor=E, padx=2)
    imgGamma.set(1.05)

    # create the image hue slider control
    hue = DoubleVar()
    sliderHue = Scale(frame2, label='Hue', variable=hue, troughcolor='blue', from_=-0.5, to=0.5, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderHue.pack(side=TOP, anchor=E, padx=2, pady=3)
    hue.set(0.0)

    # create the image saturation slider control
    saturation = DoubleVar()
    sliderSaturation = Scale(frame2, label='Saturation', variable=saturation, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSaturation.pack(side=TOP, anchor=E, padx=2)
    saturation.set(1.0)

    # create the image sharpen slider control
    sharpen = DoubleVar()
    sliderSharpen = Scale(frame2, label='Sharpen', variable=sharpen, troughcolor='blue', from_=7.9, to=9.9, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=set_sharpen_kernel)
    sliderSharpen.pack(side=TOP, padx=2, pady=3)
    sharpen.set(8.9)

    # create the image Gaussian Blur slider control
    gaussian_blur = IntVar()
    sliderGaussianBlur = Scale(frame2, label='Gaussian Blur', variable=gaussian_blur, troughcolor='blue', from_=0, to=10, resolution=2, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGaussianBlur.pack(side=TOP, padx=2)
    gaussian_blur.set(0)

    # create the image posterize slider control
    posterize = IntVar()
    sliderPosterize = Scale(frame2, label='Posterize', variable=posterize, troughcolor='blue', from_=6, to=1, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderPosterize.pack(side=TOP, padx=2, pady=3)
    posterize.set(6)

    # create the image solarize slider control
    solarize = IntVar()
    sliderSolarize = Scale(frame2, label='Solarize', variable=solarize, troughcolor='blue', from_=255, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSolarize.pack(side=TOP, padx=2)
    solarize.set(255)

    # create the image sobel threshold slider control
    sobel_threshold = IntVar()
    sliderSobelThreshold = Scale(frame2, label='Sobel Gradient', variable=sobel_threshold, troughcolor='blue', from_=0, to=4, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSobelThreshold.pack(side=TOP, padx=2, pady=3)
    sobel_threshold.set(0)

    # add 'Edges' checkbox
    show_edges = IntVar()
    chbShowEdges = Checkbutton(frame2, text='Edges', variable=show_edges, width=7, command=set_edges)
    chbShowEdges.pack(side=TOP, padx=2)
    show_edges.set(0)

    # create the image edges low threshold slider control
    low_threshold = IntVar()
    sliderLowThreshold = Scale(frame2, label='Edges Threshold', variable=low_threshold, troughcolor='blue', from_=100, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderLowThreshold.pack(side=TOP, padx=2, pady=3)
    low_threshold.set(50)

    # add 'Emboss' checkbox
    show_emboss = IntVar()
    chbShowEmboss = Checkbutton(frame2, text='Emboss', variable=show_emboss, width=7, command=set_emboss)
    chbShowEmboss.pack(side=TOP, padx=2)
    show_emboss.set(0)

    # create the image emboss slider control
    emboss = IntVar()
    sliderEmboss = Scale(frame2, label='Emboss Threshold', variable=emboss, troughcolor='blue', from_=128, to=99, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderEmboss.pack(side=TOP, padx=2, pady=3)
    emboss.set(114)

    lblScreen = Label(frame2, text='Screen', fg='grey', bg='black', font='Helvetica 9')
    lblScreen.pack(side=TOP, anchor=CENTER, pady=7)

    lblResolution = Label(frame2, text='res: ' + str(screen_width) + ' x ' + str(screen_height), fg='grey', bg='black', font='Helvetica 9')
    lblResolution.pack(side=TOP, anchor=CENTER)

    lblDPI = Label(frame2, text='dpi: ' + str(int(screenDPI)), fg='grey', bg='black', font='Helvetica 9')
    lblDPI.pack(side=TOP, anchor=CENTER)

    # add exit button
    exitBtn = Button(frame2, text='Exit', width=7, fg='red', bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=BOTTOM, anchor=CENTER, pady=5)

    lblError = Label(frame2, text='', fg='red', bg='black', font='Helvetica 12')
    lblError.pack(side=BOTTOM, anchor=CENTER, pady=10)

    #-----------------------------------------------------------------------

    # create matplotlib figure, subplot, canvas and toolbar
    fig = Figure(figsize=(640//screenDPI, 427//screenDPI), dpi=int(screenDPI))
    subplot = fig.add_subplot(111)
    subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)
    fig.set_tight_layout(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar._Spacer()
    toolbar._Button('Show Axis', None, toggle=True, command=refresh_axis)
    toolbar._Spacer()
    toolbar._Button('Reload Image', None, toggle=False, command=reload_image)
    toolbar._Spacer()
    btnFlip_H = toolbar._Button('FlipH', None, toggle=True, command=flip_image_horizontally)
    toolbar._Spacer()
    btnFlip_V = toolbar._Button('FlipV', None, toggle=True, command=flip_image_vertically)
    toolbar._Spacer()
    toolbar._Button('Histogram', None, toggle=False, command=show_histogram_window)
    toolbar.update()

    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    #-----------------------------------------------------------------------

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    show_original_image()

    root.mainloop()
