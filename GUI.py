# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
# Import everything from Tkinter module
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from os import getcwd
import os as os
import webbrowser
import math

# Scikit Learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale, LabelEncoder
import numpy as np
import pandas as pd

# Keras/Tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, Callback


# Create a Text widget that uses pixels for dimensions (width and height)
# Taken from http://code.activestate.com/recipes/578887-text-widget-width-and-height-in-pixels-tkinter/
class Text2(Frame):
    def __init__(self, master, width=0, height=0, **kwargs):
        # Width and height for frame
        self.width = width
        self.height = height

        # Initialize frame
        Frame.__init__(self, master, width=self.width, height=self.height)

        # Create text
        self.text_widget = Text(self, **kwargs)
        # Prevents text from editting
        self.text_widget.bind("<Key>", lambda e: "break")

        # Create scrollbar
        self.scroll_bar = Scrollbar(self, orient="vertical")

        # Connect text and scrollbar
        self.text_widget.configure(yscrollcommand=self.scroll_bar.set)
        self.scroll_bar.configure(command=self.text_widget.yview)

        # Pack the widgets
        self.scroll_bar.pack(side=RIGHT, fill=BOTH)
        self.text_widget.pack(expand=YES, fill=BOTH)

    def pack(self, *args, **kwargs):
        Frame.pack(self, *args, **kwargs)
        self.pack_propagate(False)

    def grid(self, *args, **kwargs):
        Frame.grid(self, *args, **kwargs)
        self.grid_propagate(False)

    def insert(self, string):
        self.text_widget.insert(END, string)
        self.text_widget.see("end")

    def delete(self):
        self.text_widget.delete(1.0, END)


# Create a Window Class
class Window:
    # Initialize PanedWindow widget
    def __init__(self):
        # Creates the root window
        self.root = Tk()

        # Creates size of the window
        self.root.geometry("1000x600")

        # Set the title of our master widget
        self.root.title("Data Science GUI")

        # Width and height for window
        self.width = 1280
        self.height = 720

        # Creates a menu instance
        menu = Menu(self.root)
        self.root.config(menu=menu)

        # Creates a file object for "File" option
        # Adds several commands to the "File" option
        fileMenu = Menu(menu, tearoff=0)
        fileMenu.add_command(label="Open File", command=self.openFile)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.client_exit)

        # Creates a help object for "Help" option
        # Adds a command to "Help" option
        helpMenu = Menu(menu, tearoff=0)
        helpMenu.add_command(label="View parameter descriptions", command=self.displayParameterDesc)

        # Adds the options to the menu
        menu.add_cascade(label="File", menu=fileMenu)
        menu.add_cascade(label="Help", menu=helpMenu)

        # Paned window for Top (main stuff) and Bottom (mainLog)
        main = PanedWindow(self.root, orient=VERTICAL, sashpad=1, sashrelief=RAISED)
        main.pack(fill=BOTH, expand=1)

        # Paned window for left (choosing features/label and parameters/algorithm) and right (displaying csv log and results log)
        top = PanedWindow(main, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)

        # Log for main stuff
        bottom = PanedWindow(main, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)

        # Adds top and bottom panedwindows
        main.add(top, height=440)
        main.add(bottom, height=200)

        # Paned Window for choosing features/label and parameters/algorithm
        left = PanedWindow(top, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)

        # LabelFrame for Main Frame
        labelFrameForMainFrame = LabelFrame(left, text="Main Frame")
        self.mainFrame = Frame(labelFrameForMainFrame)
        self.mainFrame.pack()

        # Add the label frame
        left.add(labelFrameForMainFrame)

        # Paned window for CSV File and Results
        right = PanedWindow(top, orient=VERTICAL, sashpad=1, sashrelief=RAISED)

        # Add left and right panedwindows
        top.add(left, width=500)
        top.add(right, width=300)

        # LabelFrame for CSV log
        self.labelFrameForCSVFile = LabelFrame(top, text="CSV not specified")
        # Log for CSV file
        self.csvLog = Text2(self.labelFrameForCSVFile, width=self.width, height=self.height - 300)
        self.csvLog.pack()

        # LabelFrame for Results log
        self.labelFrameForResult = LabelFrame(top, text="results not specified")
        # Log for Results
        self.resultLog = Text2(self.labelFrameForResult, width=self.width, height=self.height - 300)
        self.resultLog.pack()

        # Add the two labelframes for displaying CSV file and Result Log
        right.add(self.labelFrameForCSVFile, height=220)
        right.add(self.labelFrameForResult, height=220)

        # Show this at the beginning of the GUI
        label = Label(self.mainFrame, text="To start, click 'File' and open a CSV file.")
        label.pack()

        # Labelframe for Main log
        labelFrameForMainLog = LabelFrame(bottom, text="Main log")
        # Log for main frame
        self.mainLog = Text2(labelFrameForMainLog, width=self.width, height=self.height - 100)
        self.mainLog.insert("Started the Data Science GUI!\n")
        self.mainLog.pack()

        # Add Labelframe for main log
        bottom.add(labelFrameForMainLog)

        self.root.mainloop()

    # Open the link
    def openHyperlinkCallback(self, event):
        webbrowser.open_new(event.widget.cget("text"))

    def displayParameterDesc(self):
        # Destroy if result window and log exists.
        try:
            self.window.destroy()
            self.parameterDescLog.destroy()
        except (NameError, AttributeError):
            pass

        try:
            # Create a new frame/window from root window to display parameter descriptions
            self.window = Toplevel(self.root)
            self.window.geometry("640x300")
            self.window.title("Parameters of " + self.algorithm)

            # Display link to documentation url
            hyperlinkLabel = Label(self.window, text=self.paramDesc["link"], fg="blue", cursor="hand2")
            hyperlinkLabel.pack()
            hyperlinkLabel.bind("<Button-1>", self.openHyperlinkCallback)

            # Parameter descriptions
            self.parameterDescLog = Text2(self.window, width=self.width, height=self.height)
            self.parameterDescLog.pack()

            # Print out the results
            for key, value in self.paramDesc.items():
                if key == "link":
                    pass
                else:
                    self.parameterDescLog.insert(str(key) + ": " + str(value) + "\n\n")

        except (NameError, AttributeError):
            # Display this if user hasn't open a CSV file and select an algorithm
            label = Label(self.window,text="You need to open a CSV file and select an algorithm before displaying this!")
            label.pack()
            pass

    # Display the csv file in text
    def displayFile(self):
        self.csvLog.insert("Feature Matrix\n")
        self.csvLog.insert("----------------------\n")
        self.csvLog.insert(self.X)
        self.csvLog.insert("\n\n")
        self.csvLog.insert("Label Vector\n")
        self.csvLog.insert("----------------------\n")
        self.csvLog.insert(self.y)

    # Display results of selected algorithm and parameters
    def displayResult(self, dict):
        # Notify the user that training is done
        self.mainLog.insert("Done computing.\n")

        # Clear result log and change label frame
        self.resultLog.delete()
        self.labelFrameForResult.config(text="Results for " + self.filename + " using " + self.algorithm)

        # Print out the results
        for key, value in dict.items():
            self.resultLog.insert(str(key) + ": " + str(value) + "\n")


    def displayPredictionWindow(self):
        # Create a new frame/window from root window to display parameter descriptions
        self.predictionWindow = Toplevel(self.root)
        self.predictionWindow.geometry("480x240")
        self.predictionWindow.title("Prediction Window of " + self.algorithm)

        # Width and height for window
        width = 480
        height = 240

        # Paned window for left and right
        main = PanedWindow(self.predictionWindow, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)
        main.pack(fill=BOTH, expand=1)

        # Paned window for left  and right
        left = PanedWindow(main, orient=VERTICAL, sashpad=1, sashrelief=RAISED)

        # Log for prediction stuff
        right = PanedWindow(main, orient=VERTICAL, sashpad=1, sashrelief=RAISED)

        main.add(left, width=int(width / 2))
        main.add(right, width=int(width / 2))

        # LabelFrame for Left Frame
        labelFrameForLeftFrame = LabelFrame(left, text="Prediction Frame")

        # Object containing feature columns and their datatypes
        self.datatypes = self.X.dtypes

        # Validation command
        # %d = Type of action (1=insert, 0=delete, -1 for others)
        # %P = value of the entry if the edit is allowed (all, focusin, focusout, forced)
        vcmdForInt = (self.predictionWindow.register(self.validateInt2),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat = (self.predictionWindow.register(self.validateFloat),
                        '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat2 = (self.predictionWindow.register(self.validateFloat2),
                         '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        # Contains entries
        arrayOfEntries = []

        # Counter for row
        counterForRow = 0

        # Go through an object that contains the feature columns and their datatypes
        for attr, value in self.datatypes.items():
            Label(labelFrameForLeftFrame, text=attr).grid(row=counterForRow, sticky=W)

            entry = None
            if (value == "int64"):
                entry = Entry(labelFrameForLeftFrame, width=30, validate="all", validatecommand=vcmdForInt)
                entry.insert(0, 1)
            elif (value == "float64"):
                entry = Entry(labelFrameForLeftFrame, width=30, validate="all", validatecommand=vcmdForFloat2)
                entry.insert(0, 1.0)

            entry.grid(row=counterForRow, column=1, sticky=E)
            arrayOfEntries.append(entry)
            counterForRow = counterForRow + 1

        button = Button(labelFrameForLeftFrame, text="Predict", command=lambda: self.predict(arrayOfEntries))
        button.grid(row=counterForRow, columnspan=2, sticky=W + E)

        left.add(labelFrameForLeftFrame, width=int(width / 2), height=height)

        # LabelFrame for Right Frame
        labelFrameForRightFrame = LabelFrame(right, text="Prediction Frame")
        # Log
        self.predictionLog = Text2(labelFrameForRightFrame, width=int(width / 2), height=height)
        self.predictionLog.pack()

        right.add(labelFrameForRightFrame, width=int(width / 2), height=height)

    # For user-input predictions
    def predict(self, entries):
        for entry in entries:
            if not entry.get().strip():
                self.predictionWindow.bell()
                self.mainLog.insert("Check the parameters! You may have entered nothing or 0 for an input!\n")
                return

        counter = 0
        arrayForUserInput = []
        for attr, value in self.datatypes.items():
            if (value == "int64"):
                arrayForUserInput.append(int(entries[counter].get()))
            elif (value == "float64"):
                arrayForUserInput.append(float(entries[counter].get()))

            self.predictionLog.insert(attr + ": " + entries[counter].get() + "\n")

            counter = counter + 1

        arrayForUserInput = np.array(arrayForUserInput).reshape(1, -1)

        prediction = None

        if (self.algorithm != "Keras"):
            prediction = self.classifier.predict(arrayForUserInput)[0]
        else:
            # Using label_encoder to reverse the one-hot-encoding (from numerical label to categorical label)
            prediction = self.label_encoder.inverse_transform(np.argmax(self.classifier.predict(arrayForUserInput)[0]))
        self.predictionLog.insert("Prediction: " + str(prediction) + "\n\n")


    # Remove features from label listbox when user selects the features
    def removeFeatures(self, event):
        # Note here that Tkinter passes an event object
        w = event.widget

        # Indexes of selected features
        indexes = w.curselection()

        # Columns of dataframe
        cols = list(self.df.columns.values)

        selected_features = [cols[item] for item in indexes]

        self.list_of_label.delete(0, END)

        for col in cols:
            if col not in selected_features:
                self.list_of_label.insert(0, col)

    # Prompts the user to open a CSV File
    def openFile(self):
        # The full path of the file
        file = filedialog.askopenfilename(initialdir=getcwd() + "/csv", title="Select file",
                                          filetypes=(("csv files", "*.csv"),))

        if file:
            # Actual filename
            self.filename = os.path.basename(file)

            # Notify user that program is reading off the csv
            self.mainLog.insert("Reading '" + self.filename + "' from '" + file + "'.\n")

            # Clear the selection_Frame
            # Clear the widgets in the selection frame after you selected features and labels
            for widget in self.mainFrame.winfo_children():
                widget.destroy()

            # Change LabelFrame text to filename
            self.labelFrameForCSVFile.config(text=self.filename)

            # Clear the CSV log
            self.csvLog.delete()

            # Clear Results log
            self.resultLog.delete()

            # Dataframe created from the file
            self.df = pd.read_csv(file, sep=',')

            # Columns of dataframe
            cols = list(self.df.columns.values)

            # Create a listbox
            self.list_of_features = Listbox(self.mainFrame, selectmode=MULTIPLE, width=30, height=10, exportselection=0)
            self.list_of_features.bind('<<ListboxSelect>>', self.removeFeatures)
            self.list_of_label = Listbox(self.mainFrame, selectmode=SINGLE, width=30, height=10, exportselection=0)

            # Show a list of columns for user to check
            for column in cols:
                self.list_of_features.insert(END, column)
                self.list_of_label.insert(END, column)

            # Display label, listbox, and button
            Label(self.mainFrame, text="Select the feature columns", relief=RIDGE).pack()
            self.list_of_features.pack()

            Label(self.mainFrame, text="Select the label", relief=RIDGE).pack()
            self.list_of_label.pack()

            ok = Button(self.mainFrame, text="Okay", command=self.setUpMatrixes)
            ok.pack()

    # Set up the feature matrix and label vector
    def setUpMatrixes(self):
        # The selections of feature columns and labels
        columns = self.list_of_features.get(0, END)
        indexesForFeatureCols = self.list_of_features.curselection()
        selected_features = [columns[item] for item in indexesForFeatureCols]
        selected_label = self.list_of_label.get(ANCHOR)

        # Clear the widgets in the selection frame after you selected features and labels
        for widget in self.mainFrame.winfo_children():
            widget.destroy()

        # Notify user of selected features and label
        self.mainLog.insert("You have selected " + str(selected_features) + " as features.\n")
        self.mainLog.insert("You have selected " + str(selected_label) + " as the label.\n")

        # Feature matrix and label vector
        self.X = self.df[selected_features]
        self.y = self.df[selected_label]

        # Labels
        self.labels = self.df[selected_label].unique()

        # Number of features and labels
        self.numberOfFeatures = len(selected_features)
        self.numberOfLabels = len(self.labels)

        # Option Menu for choosing machine learning algorithms
        algorithms = ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Linear Regression",
                      "Logistic Regression", "Linear SVC", "Multilayer Perceptron", "Keras"]
        self.default = StringVar()
        self.default.set("Select an algorithm.")
        self.options = OptionMenu(self.mainFrame, self.default, *algorithms, command=self.selectedAlgorithm)
        self.options.pack()

        # Display the csv file
        self.displayFile()

    # Display parameters for the selected algorithm
    def selectedAlgorithm(self, algorithm):
        # Clear the widgets in the selection frame when changing algorithm
        for widget in self.mainFrame.winfo_children():
            widget.destroy()

        # Option Menu for choosing machine learning algorithms
        algorithms = ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Linear Regression",
                      "Logistic Regression", "Linear SVC", "Multilayer Perceptron", "Keras"]
        self.default = StringVar()
        self.default.set(algorithm)
        self.options = OptionMenu(self.mainFrame, self.default, *algorithms, command=self.selectedAlgorithm)
        self.options.pack()

        # Validation command
        # %d = Type of action (1=insert, 0=delete, -1 for others)
        # %P = value of the entry if the edit is allowed (all, focusin, focusout, forced)
        vcmdForInt = (self.mainFrame.register(self.validateInt),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat = (self.mainFrame.register(self.validateFloat),
                        '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat2 = (self.mainFrame.register(self.validateFloat2),
                         '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForHiddenLayerSizes = (self.mainFrame.register(self.validateHiddenLayerSizes),
                                   '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        self.algorithm = algorithm

        # Load image
        load = Image.open("img/" + algorithm + ".png")

        # Load image
        load = load.resize((100, 100), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # Labels can be text or images
        img = Label(self.mainFrame, image=render)
        img.image = render
        img.pack()

        self.paramDesc = {}
        parameters = []

        Label(self.mainFrame, text="test_size", relief=RIDGE).pack()
        self.test_size = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat)
        self.test_size.insert(0, 0.3)
        self.test_size.pack()
        parameters.append(self.test_size)

        Label(self.mainFrame, text="random_state", relief=RIDGE).pack()
        self.random_state = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
        self.random_state.insert(0, 2)
        self.random_state.pack()
        parameters.append(self.random_state)

        Label(self.mainFrame, text="cv", relief=RIDGE).pack()
        self.cv = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
        self.cv.insert(0, 10)
        self.cv.pack()
        parameters.append(self.cv)

        self.paramDesc[
            "test_size"] = "float, int or None, optional (default=0.25)\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. By default, the value is set to 0.25. The default will change in version 0.21. It will remain 0.25 only if train_size is unspecified, otherwise it will complement the specified train_size."
        self.paramDesc[
            "random_state"] = "int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random."
        self.paramDesc[
            "cv"] = " int, cross-validation generator or an iterable, optional\nDetermines the cross-validation splitting strategy. Possible inputs for cv are:\nNone, to use the default 3-fold cross validation,\ninteger, to specify the number of folds in a (Stratified)KFold,\nAn object to be used as a cross-validation generator.\nAn iterable yielding train, test splits."

        if self.algorithm == "K-Nearest Neighbors":
            Label(self.mainFrame, text="n_neighbors", relief=RIDGE).pack()
            self.n_neighbors = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.n_neighbors.insert(0, 5)
            self.n_neighbors.pack()
            parameters.append(self.n_neighbors)

            self.paramDesc[
                "link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"
            self.paramDesc[
                "n_neighbors"] = "int, optional (default = 5)\nNumber of neighbors to use by default for kneighbors queries."

        elif algorithm == "Decision Tree":
            self.paramDesc[
                "link"] = "https////scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"

        elif algorithm == "Random Forest":
            Label(self.mainFrame, text="n_estimators", relief=RIDGE).pack()
            self.n_estimators = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.n_estimators.insert(0, 19)
            self.n_estimators.pack()
            parameters.append(self.n_estimators)

            self.paramDesc[
                "link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
            self.paramDesc["n_estimators"] = "integer, optional (default=10)\nThe number of trees in the forest."

        elif algorithm == "Linear Regression":
            self.paramDesc[
                "link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"

        elif algorithm == "Logistic Regression":
            self.paramDesc[
                "link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"

        elif algorithm == "Linear SVC":
            Label(self.mainFrame, text="C", relief=RIDGE).pack()
            self.c = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat2)
            self.c.insert(0, 1.0)
            self.c.pack()
            parameters.append(self.c)

            Label(self.mainFrame, text="max_iter", relief=RIDGE).pack()
            self.max_iter = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.max_iter.insert(0, 100)
            self.max_iter.pack()
            parameters.append(self.max_iter)

            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html"
            self.paramDesc["C"] = "float, optional (default=1.0)\nPenalty parameter C of the error term."
            self.paramDesc["max_iter"] = "int, (default=1000)\nThe maximum number of iterations to be run."

        elif algorithm == "Multilayer Perceptron":
            Label(self.mainFrame, text="max_iter", relief=RIDGE).pack()
            self.max_iter = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.max_iter.insert(0, 100)
            self.max_iter.pack()
            parameters.append(self.max_iter)

            Label(self.mainFrame, text="alpha", relief=RIDGE).pack()
            self.alpha = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat2)
            self.alpha.insert(0, 0.005)
            self.alpha.pack()
            parameters.append(self.alpha)

            Label(self.mainFrame, text="hidden_layer_sizes", relief=RIDGE).pack()
            self.hidden_layer_sizes = Entry(self.mainFrame, validate="all", validatecommand=vcmdForHiddenLayerSizes)
            self.hidden_layer_sizes.insert(0, 2)
            self.hidden_layer_sizes.pack()
            parameters.append(self.hidden_layer_sizes)

            self.paramDesc[
                "link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"
            self.paramDesc["max_iter"] = "int, (default=1000)\nThe maximum number of iterations to be run."
            self.paramDesc["alpha"] = "float, optional, default 0.0001\nL2 penalty (regularization term) parameter."
            self.paramDesc[
                "hidden_layer_sizes"] = "tuple, length = n_layers - 2, default (100,)\nThe ith element represents the number of neurons in the ith hidden layer."

        elif algorithm == "Keras":
            Label(self.mainFrame, text="Number of hidden layers", relief=RIDGE).pack()
            self.numberOfHiddenLayers = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.numberOfHiddenLayers.insert(0, 1)
            self.numberOfHiddenLayers.pack()
            parameters.append(self.numberOfHiddenLayers)

            Label(self.mainFrame, text="Hidden layer sizes", relief=RIDGE).pack()
            self.hidden_layer_sizes = Entry(self.mainFrame, validate="all", validatecommand=vcmdForHiddenLayerSizes)
            self.hidden_layer_sizes.insert(0, 2)
            self.hidden_layer_sizes.pack()
            parameters.append(self.hidden_layer_sizes)

            Label(self.mainFrame, text="epochs", relief=RIDGE).pack()
            self.epochs = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.epochs.insert(0, 200)
            self.epochs.pack()
            parameters.append(self.epochs)

            Label(self.mainFrame, text="batch_size", relief=RIDGE).pack()
            self.batch_size = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.batch_size.insert(0, 128)
            self.batch_size.pack()
            parameters.append(self.batch_size)

            self.paramDesc["link"] = "https:////keras.io/models/sequential/"
            self.paramDesc["number of hidden layers"] = "Integer.\nSpecify the number of hidden layers."
            self.paramDesc[
                "hidden layer sizes"] = "tuple, length = n_layers - 2, default (100,)\nThe ith element represents the number of neurons in the ith hidden layer."
            self.paramDesc[
                "epochs"] = "Integer.\nNumber of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch,  epochs is to be understood as 'final epoch'. The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached."
            self.paramDesc[
                "batch_size"] = " Integer or None.\nNumber of samples per gradient update. If unspecified, batch_size will default to 32."

        # Compute using the specified parameters
        # Lambda means that the method won't be called immediately (only when button is pressed)
        submit = Button(self.mainFrame, text="Submit", command=lambda: self.validateAllInputs(parameters))
        submit.pack()

        # Notify user that program is reading off the csv
        self.mainLog.insert(self.algorithm + " has been selected!\n")

    # Validate integer inputs (don't allow user to enter anything else)
    def validateInt(self, d, i, P, s, S, v, V, W):
        # print("end", "OnValidate:\n")
        # print("end", "d='%s'\n" % d)
        # print("end", "i='%s'\n" % i)
        # print("end", "P='%s'\n" % P)
        # print("end", "s='%s'\n" % s)
        # print("end", "S='%s'\n" % S)
        # print("end", "v='%s'\n" % v)
        # print("end", "V='%s'\n" % V)
        # print("end", "W='%s'\n" % W)

        # Accept Integer values and empty string (for erasing the one only number)
        if (P.isdigit() and int(P) > 0) or P == "":
            return True
        else:
            self.mainLog.insert("Please enter an integer (if the field is empty, enter an integer greater than 0).\n")
            self.mainFrame.bell()
            return False


    # Validate integer inputs (don't allow user to enter anything else)
    def validateInt2(self, d, i, P, s, S, v, V, W):
        # Accept Integer values and empty string (for erasing the one only number)
        if (P.isdigit() and int(P) >= 0) or P == "":
            return True
        else:
            self.mainLog.insert("Please enter an integer (if the field is empty, enter an integer greater than 0).\n")
            self.mainFrame.bell()
            return False

    # Validate float inputs (don't allow user to enter anything else)
    def validateFloat(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True
        elif S == " ":
            self.mainLog.insert("No spaces! Enter a digit!\n")
            self.mainFrame.bell()
            return False

        try:
            number = float(P)

            if (0.0 <= number and number <= 1.0):
                return True
            else:
                self.mainLog.insert("Float numbers must be between 0.0 and 1.0 (inclusive)!\n")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.insert("Float numbers are only allowed (ex: 0.3 or .3)!\n")
            self.mainFrame.bell()
            return False

    # Vaidate float inputs (don't allow user to enter anything else)
    def validateFloat2(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True
        elif S == " ":
            self.mainLog.insert("No spaces! Enter a digit!\n")
            self.mainFrame.bell()
            return False

        try:
            number = float(P)

            if (0.0 <= number and number <= 1000.0):
                return True
            else:
                self.mainLog.insert("Float numbers must be between 0.00001 and 1000.0 (inclusive)!\n")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.insert("Float numbers are only allowed (ex: 0.00001 or .00001)!\n")
            self.mainFrame.bell()
            return False

    # Vaidate hidden layer sizes inputs (don't allow user to enter anything else)
    def validateHiddenLayerSizes(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            if S.isdigit() or S == "," or S == "":
                return True
            else:
                self.mainLog.insert("Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.insert("Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
            self.mainFrame.bell()
            return False

    # Final validation of inputs (doesn't compute anything if a parameter is field or is 0
    def validateAllInputs(self, parameters):
        for parameter in parameters:
            if not parameter.get().strip() or parameter.get() == "0":
                self.mainFrame.bell()
                self.mainLog.insert("Check the parameters! You may have entered nothing or 0 for an input!\n")
                return

        self.compute()

    # Compute the results of
    def compute(self):
        self.mainLog.insert("Computing...\n")
        self.classifier = None

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.test_size.get()),
                                                            random_state=int(self.random_state.get()))

        if self.algorithm == "K-Nearest Neighbors":
            # Instantiating KNN object
            self.classifier = KNeighborsClassifier(n_neighbors=int(self.n_neighbors.get()))
        elif self.algorithm == "Decision Tree":
            # Instantiating DecisionTreeClassifier object
            self.classifier = DecisionTreeClassifier()
        elif self.algorithm == "Random Forest":
            # Instantiating RandomForestClassifier object
            self.classifier = RandomForestClassifier(n_estimators=int(self.n_estimators.get()), bootstrap=True)
        elif self.algorithm == "Linear Regression":
            # Instantiating Linear Regression object
            self.classifier = LinearRegression()
        elif self.algorithm == "Logistic Regression":
            # Instantiating Logistic Regression object
            self.classifier = LogisticRegression()
        elif self.algorithm == "Linear SVC":
            # LinearSVC classifier
            self.classifier = LinearSVC()
        elif self.algorithm == "Multilayer Perceptron":
            # Turn the string (containing commas) into a list
            modified_hidden_layer_sizes = self.hidden_layer_sizes.get().split(",")

            # Remove any empty string in the list
            modified_hidden_layer_sizes = [item.strip() for item in modified_hidden_layer_sizes if item.strip()]

            # Turn the list of strings into a tuple of int
            modified_hidden_layer_sizes = tuple([int(i) for i in modified_hidden_layer_sizes])

            # Instantiate MLP classifier
            self.classifier = MLPClassifier(activation='logistic', solver='adam', max_iter=int(self.max_iter.get()),
                                            learning_rate_init=0.002, alpha=float(self.alpha.get()),
                                            hidden_layer_sizes=modified_hidden_layer_sizes)

        elif self.algorithm == "Keras":
            # Make numpy arrays from the split dataframes
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Label encoder for transforming categorical labels into numerical labels (one-hot encoded)
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            y_train = pd.get_dummies(y_train).values

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            y_test = self.label_encoder.fit_transform(y_test)
            y_test = pd.get_dummies(y_test).values

            # Turn the string (containing commas) into a list
            modified_hidden_layer_sizes = self.hidden_layer_sizes.get().split(",")

            # Remove any empty string in the list
            modified_hidden_layer_sizes = [item.strip() for item in modified_hidden_layer_sizes if item.strip()]

            # Turn the list of strings into a tuple of int
            modified_hidden_layer_sizes = tuple([int(i) for i in modified_hidden_layer_sizes])

            # create model
            self.classifier = Sequential()
            # Input layer (input is the number of features we have)
            self.classifier.add(Dense(self.numberOfFeatures, input_dim=self.numberOfFeatures, activation='relu'))
            # Normalize the activations of the previous layer at each batch
            self.classifier.add(BatchNormalization())

            # Hidden Layer (number of features + number of labels)/2 is the rule of thumb for neurons in hidden layer
            for i in modified_hidden_layer_sizes:
                self.classifier.add(Dense(i, activation='relu'))

            # Output layer
            self.classifier.add(Dense(self.numberOfLabels, activation='softmax'))
            # model.add(Dropout(0.2))

            # Compile model
            self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Print epoch number
            epoch_begin_print_callback = LambdaCallback(
                on_epoch_begin=lambda epoch, logs: self.mainLog.insert(str(epoch) + "\n"))

            # Print the batch number at the beginning of every batch.
            batch_print_callback = LambdaCallback(
                on_batch_begin=lambda batch, logs: self.mainLog.insert(str(batch) + "\n"))

            # Fit the model
            self.classifier.fit(X_train, y_train, epochs=int(self.epochs.get()), batch_size=int(self.batch_size.get()))

            # Accuracy of testing data on predictive model
            score = self.classifier.evaluate(X_test, y_test, batch_size=int(self.batch_size.get()))

        if (self.algorithm != "Keras"):
            # fit the model with the training set
            self.classifier.fit(X_train, y_train)

            # Predict method is used for creating a prediction on testing data
            y_predict = self.classifier.predict(X_test)

            # Accuracy of testing data on predictive model
            accuracy = accuracy_score(y_test, y_predict)

            # Add #-fold Cross Validation with Supervised Learning
            accuracy_list = cross_val_score(self.classifier, self.X, self.y, cv=int(self.cv.get()), scoring='accuracy')

            # # Report
            # report = classification_report(y_test, y_predict, target_names=self.labels)

            # Dictionary containing information
            dict = {"Training Set Size": 1.00 - float(self.test_size.get()),
                    "Testing Set Size": float(self.test_size.get()),
                    "Training Set Shape": X_train.shape,
                    "Testing Set Shape": X_test.shape,
                    "Classifier": self.classifier,
                    "Accuracy for " + self.algorithm: str(accuracy),
                    "Cross Validation for " + self.algorithm: accuracy_list.mean()}

        else:
            # Dictionary containing information
            dict = {"Training Set Size": 1.00 - float(self.test_size.get()),
                    "Testing Set Size": float(self.test_size.get()),
                    "Training Set Shape": X_train.shape,
                    "Testing Set Shape": X_test.shape,
                    "Number Of hidden layers": self.numberOfHiddenLayers.get(),
                    "Hidden layer sizes:": self.hidden_layer_sizes.get(),
                    "Epochs": self.epochs.get(),
                    "Batch Size": self.batch_size.get(),
                    "Loss": score[0],
                    "Accuracy": score[1],
                    }

        self.displayResult(dict)
        self.displayPredictionWindow()

    # Exit the client
    def client_exit(self):
        exit()


def main():
    # Create an instance of window
    app = Window()


if __name__ == '__main__':
    main()
