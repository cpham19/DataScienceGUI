# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
#Import everything from Tkinter module
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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Keras/Tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical

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
        self.root.config(menu = menu)

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

        # Adds top and bototm panedwindows
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
        self.mainLog.text_widget.insert(END, "Started the Data Science GUI!\n")
        self.mainLog.text_widget.see("end")
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
                    self.parameterDescLog.text_widget.insert(END, str(key) + ": " + str(value) + "\n")
                    self.parameterDescLog.text_widget.insert(END, "\n")

            self.parameterDescLog.text_widget.see("end")

        except (NameError, AttributeError):
            # Display this if user hasn't open a CSV file and select an algorithm
            label = Label(self.window, text="You need to open a CSV file and select an algorithm before displaying this!")
            label.pack()
            pass

    # Display the csv file in text
    def displayFile(self):
        self.csvLog.text_widget.insert(END, "Feature Matrix\n")
        self.csvLog.text_widget.insert(END, "----------------------\n")
        self.csvLog.text_widget.insert(END, self.X)
        self.csvLog.text_widget.insert(END, "\n\n")
        self.csvLog.text_widget.insert(END, "Label Vector\n")
        self.csvLog.text_widget.insert(END, "----------------------\n")
        self.csvLog.text_widget.insert(END, self.y)
        self.csvLog.text_widget.see("end")

    # Display results of selected algorithm and parameters
    def displayResult(self, dict):
        # Notify the user that training is done
        self.mainLog.text_widget.insert(END, "Done computing.\n")
        self.mainLog.text_widget.see("end")

        # Clear result log and change label frame
        self.resultLog.delete()
        self.labelFrameForResult.config(text="Results for " + self.filename + " using " + self.algorithm)

        # Print out the results
        for key, value in dict.items():
            self.resultLog.text_widget.insert(END, str(key) + ": " + str(value) + "\n")

        self.resultLog.text_widget.see("end")

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
        file = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("csv files","*.csv"),))

        if file:
            # Actual filename
            self.filename = os.path.basename(file)

            # Notify user that program is reading off the csv
            self.mainLog.text_widget.insert(END, "Reading '" + self.filename + "' from '" + file + "'.\n")
            self.mainLog.text_widget.see("end")

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
        self.mainLog.text_widget.insert(END, "You have selected " + str(selected_features) + " as features.\n")
        self.mainLog.text_widget.insert(END, "You have selected " + str(selected_label) + " as the label.\n")
        self.mainLog.text_widget.see("end")

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
        load = Image.open(algorithm + ".png")

        # Load image
        load = load.resize((100, 100), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # Labels can be text or images
        img = Label(self.mainFrame, image=render)
        img.image = render
        img.pack()

        self.paramDesc = {}

        if (self.algorithm != "Keras"):
            Label(self.mainFrame, text="test_size", relief=RIDGE).pack()
            self.test_size = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat)
            self.test_size.insert(0, 0.3)
            self.test_size.pack()

            self.paramDesc["test_size"] = "float, int or None, optional (default=0.25)\nIf float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. By default, the value is set to 0.25. The default will change in version 0.21. It will remain 0.25 only if train_size is unspecified, otherwise it will complement the specified train_size."

            Label(self.mainFrame, text="random_state", relief=RIDGE).pack()
            self.random_state = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.random_state.insert(0, 2)
            self.random_state.pack()

            self.paramDesc["random_state"] = "int, RandomState instance or None, optional (default=None)\nIf int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random."

            Label(self.mainFrame, text="cv", relief=RIDGE).pack()
            self.cv = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.cv.insert(0, 10)
            self.cv.pack()

            self.paramDesc["cv"] = " int, cross-validation generator or an iterable, optional\nDetermines the cross-validation splitting strategy. Possible inputs for cv are:\nNone, to use the default 3-fold cross validation,\ninteger, to specify the number of folds in a (Stratified)KFold,\nAn object to be used as a cross-validation generator.\nAn iterable yielding train, test splits."


        if self.algorithm == "K-Nearest Neighbors":
            Label(self.mainFrame, text="n_neighbors", relief=RIDGE).pack()
            self.n_neighbors = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.n_neighbors.insert(0, 5)
            self.n_neighbors.pack()

            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"
            self.paramDesc["n_neighbors"] = "int, optional (default = 5)\nNumber of neighbors to use by default for kneighbors queries."

        elif algorithm == "Decision Tree":
            self.paramDesc["link"] = "https////scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"

        elif algorithm == "Random Forest":
            Label(self.mainFrame, text="n_estimators", relief=RIDGE).pack()
            self.n_estimators = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.n_estimators.insert(0, 19)
            self.n_estimators.pack()

            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
            self.paramDesc["n_estimators"] = "integer, optional (default=10)\nThe number of trees in the forest."

        elif algorithm == "Linear Regression":
            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"

        elif algorithm == "Logistic Regression":
            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"

        elif algorithm == "Linear SVC":
            Label(self.mainFrame, text="C", relief=RIDGE).pack()
            self.c = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat2)
            self.c.insert(0, 1.0)
            self.c.pack()

            Label(self.mainFrame, text="max_iter", relief=RIDGE).pack()
            self.max_iter = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.max_iter.insert(0, 100)
            self.max_iter.pack()

            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html"
            self.paramDesc["C"] = "float, optional (default=1.0)\nPenalty parameter C of the error term."
            self.paramDesc["max_iter"] = "int, (default=1000)\nThe maximum number of iterations to be run."

        elif algorithm == "Multilayer Perceptron":
            Label(self.mainFrame, text="max_iter", relief=RIDGE).pack()
            self.max_iter = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.max_iter.insert(0, 100)
            self.max_iter.pack()

            Label(self.mainFrame, text="alpha", relief=RIDGE).pack()
            self.alpha = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat2)
            self.alpha.insert(0, 0.005)
            self.alpha.pack()

            Label(self.mainFrame, text="hidden_layer_sizes", relief=RIDGE).pack()
            self.hidden_layer_sizes = Entry(self.mainFrame, validate="all", validatecommand=vcmdForHiddenLayerSizes)
            self.hidden_layer_sizes.insert(0, 2)
            self.hidden_layer_sizes.pack()

            self.paramDesc["link"] = "https:////scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"
            self.paramDesc["max_iter"] = "int, (default=1000)\nThe maximum number of iterations to be run."
            self.paramDesc["alpha"] = "float, optional, default 0.0001\nL2 penalty (regularization term) parameter."
            self.paramDesc["hidden_layer_sizes"] = "tuple, length = n_layers - 2, default (100,)\nThe ith element represents the number of neurons in the ith hidden layer."

        elif algorithm == "Keras":
            Label(self.mainFrame, text="epochs", relief=RIDGE).pack()
            self.epochs = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.epochs.insert(0, 200)
            self.epochs.pack()

            Label(self.mainFrame, text="batch_size", relief=RIDGE).pack()
            self.batch_size = Entry(self.mainFrame, validate="all", validatecommand=vcmdForInt)
            self.batch_size.insert(0, 128)
            self.batch_size.pack()

            Label(self.mainFrame, text="validation_split", relief=RIDGE).pack()
            self.validation_split = Entry(self.mainFrame, validate="all", validatecommand=vcmdForFloat)
            self.validation_split.insert(0, 0.1)
            self.validation_split.pack()

            self.paramDesc["link"] = "https:////keras.io/models/sequential/"
            self.paramDesc["epochs"] = "Integer.\nNumber of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch,  epochs is to be understood as 'final epoch'. The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached."
            self.paramDesc["batch_size"] = " Integer or None.\nNumber of samples per gradient update. If unspecified, batch_size will default to 32."
            self.paramDesc['validation_split'] = "Float between 0 and 1.\nFraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling."

        # Compute using the specified parameters
        submit = Button(self.mainFrame, text="Submit", command=self.compute)
        submit.pack()

        # Notify user that program is reading off the csv
        self.mainLog.text_widget.insert(END, self.algorithm + " has been selected!\n")
        self.mainLog.text_widget.see("end")

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
        if P.isdigit() or P == "":
            return True
        else:
            self.mainLog.text_widget.insert(END, "Please enter an integer.\n")
            self.mainLog.text_widget.see("end")
            self.mainFrame.bell()
            return False

    # Validate float inputs (don't allow user to enter anything else)
    def validateFloat(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            number = float(P)

            if (0.0 <= number and number <= 1.0):
                return True
            else:
                self.mainLog.text_widget.insert(END, "Float numbers must be between 0.0 and 1.0 (inclusive)!\n")
                self.mainLog.text_widget.see("end")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Float numbers are only allowed (ex: 0.3 or .3)!\n")
            self.mainLog.text_widget.see("end")
            self.mainFrame.bell()
            return False

    # Vaidate float inputs (don't allow user to enter anything else)
    def validateFloat2(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            number = float(P)

            if (0.0 <= number and number <= 1000.0):
                return True
            else:
                self.mainLog.text_widget.insert(END, "Float numbers must be between 0.00001 and 1000.0 (inclusive)!\n")
                self.mainLog.text_widget.see("end")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Float numbers are only allowed (ex: 0.00001 or .00001)!\n")
            self.mainLog.text_widget.see("end")
            self.mainFrame.bell()
            return False

    # Vaidate hidden layer sizes inputs (don't allow user to enter anything else)
    def validateHiddenLayerSizes(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            hidden_layer_sizes = P.split(",")

            if S.isdigit() or S == "," or S == "" :
                return True
            else:
                self.mainLog.text_widget.insert(END, "Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
                self.mainLog.text_widget.see("end")
                self.mainFrame.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
            self.mainLog.text_widget.see("end")
            self.mainFrame.bell()
            return False

    # Compute the results of classifier
    def compute(self):
        self.mainLog.text_widget.insert(END, "Computing...\n")
        self.mainLog.text_widget.see("end")

        classifier = None

        if (self.algorithm != "Keras"):
            # Split the dataframe dataset.
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(0.3),
                                                                random_state=int(self.random_state.get()))

            if self.algorithm == "K-Nearest Neighbors":
                # Instantiating KNN object
                classifier = KNeighborsClassifier(n_neighbors=int(self.n_neighbors.get()))
            elif self.algorithm == "Decision Tree":
                # Instantiating DecisionTreeClassifier object
                classifier = DecisionTreeClassifier()
            elif self.algorithm == "Random Forest":
                # Instantiating RandomForestClassifier object
                classifier = RandomForestClassifier(n_estimators=int(self.n_estimators.get()), bootstrap=True)
            elif self.algorithm == "Linear Regression":
                # Instantiating Linear Regression object
                classifier = LinearRegression()
            elif self.algorithm == "Logistic Regression":
                # Instantiating Logistic Regression object
                classifier = LogisticRegression()
            elif self.algorithm == "Linear SVC":
                # LinearSVC classifier
                classifier = LinearSVC()
            elif self.algorithm == "Multilayer Perceptron":
                # Turn the string (containing commas) into a list
                modified_hidden_layer_sizes = self.hidden_layer_sizes.get().split(",")

                # Remove any empty string in the list
                modified_hidden_layer_sizes = [item.strip() for item in modified_hidden_layer_sizes if item.strip()]

                # Turn the list of strings into a tuple of int
                modified_hidden_layer_sizes = tuple([int(i) for i in modified_hidden_layer_sizes])

                # Instantiate MLP classifier
                classifier = MLPClassifier(activation= 'logistic', solver='adam',max_iter=int(self.max_iter.get()),
                                           learning_rate_init=0.002, alpha=float(self.alpha.get()), hidden_layer_sizes=modified_hidden_layer_sizes)


            # fit the model with the training set
            classifier.fit(X_train, y_train)

            # Predict method is used for creating a prediction on testing data
            y_predict = classifier.predict(X_test)

            # Accuracy of testing data on predictive model
            accuracy = accuracy_score(y_test, y_predict)

            # Add #-fold Cross Validation with Supervised Learning
            accuracy_list = cross_val_score(classifier, self.X, self.y, cv=int(self.cv.get()), scoring='accuracy')

            # # Report
            # report = classification_report(y_test, y_predict, target_names=self.labels)

            # Dictionary containing information
            dict = {"Training Set Size": 1.00 - float(self.test_size.get()),
                    "Testing Set Size": float(self.test_size.get()),
                    "Training Set Shape": X_train.shape,
                    "Testing Set Shape": X_test.shape,
                    "Classifier": classifier,
                    "Accuracy for " + self.algorithm : str(accuracy),
                    "Cross Validation for " + self.algorithm : accuracy_list.mean()}

        elif self.algorithm == "Keras":
            # Make numpy arrays from X and y dataframe
            X = np.array(self.X)
            y = np.array(self.y)
            y1 = LabelEncoder().fit_transform(y)
            Y = pd.get_dummies(y1).values

            # create model
            model = Sequential()
            # Input layer (input is the number of features we have)
            model.add(Dense(self.numberOfFeatures, input_dim=self.numberOfFeatures, activation='relu'))
            # Normalize the activations of the previous layer at each batch
            model.add(BatchNormalization())
            # Hidden Layer (number of features + number of labels)/2 is the rule of thumb for neurons in hidden layer
            model.add(Dense(math.ceil((self.numberOfFeatures + self.numberOfLabels) / 2), activation='relu'))
            # Output layer
            model.add(Dense(self.numberOfLabels, activation='softmax'))
            # model.add(Dropout(0.2))

            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Fit the model
            history = model.fit(X, Y, epochs=int(self.epochs.get()), batch_size=int(self.batch_size.get()), validation_split=float(self.validation_split.get())).history

            # Dictionary containing information
            dict = {"Epochs": self.epochs.get(),
                    "Batch Size": self.batch_size.get(),
                    "Accuracy": str(np.mean(history['acc']))
                    }


        self.displayResult(dict)


    # Exit the client
    def client_exit(self):
        exit()


def main():
    # Create an instance of window
    app = Window()


if __name__ == '__main__':
    main()
