# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
#Import everything from Tkinter module
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import messagebox
from os import getcwd
import os as os
import csv
import math

# Scikit Learn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
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


# Create a class that inherits from Frame class
class Window(Frame):

    # Initialize PanedWindow widget
    def __init__(self, master=None):
        # Parameters sent to Frame class
        PanedWindow.__init__(self, master)

        # Reference to master widget (tk window)
        self.master = master

        # Run
        self.init_window()

    # Initialize window
    def init_window(self):
        # Width and height for window
        self.width = 800
        self.height = 640

        # Set the title of our master widget
        self.master.title("Data Science GUI")

        # Allows the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # Adds a textbox at the bottom
        # Height is the lines to show, width is the number of characters to show
        self.mainLog = Text2(self.master, width=self.width, height=self.height - 500)
        self.mainLog.text_widget.insert(END, "Started the Data Science GUI!\n")
        self.mainLog.text_widget.see("end")
        self.mainLog.pack()

        # Creates a menu instance
        menu = Menu(self.master)
        self.master.config(menu = menu)

        # Creates a file object for "File" option
        # Adds several commands to the "File" option
        fileMenu = Menu(menu, tearoff=0)
        fileMenu.add_command(label="Open File", command=self.openFile)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.client_exit)

        # Adds the options to the menu
        menu.add_cascade(label="File", menu=fileMenu)


    # Display the csv file in text
    def displayFile(self):
        # Create a new frame/window from root window to display CSV file
        self.window = Toplevel(self)
        self.window.geometry("640x300")
        self.window.title(self.filename)

        # Adds a textbox
        # Height is the lines to show, width is the number of characters to show
        self.csvLog = Text2(self.window, width=self.width, height=self.height)
        self.csvLog.text_widget.insert(END, "Feature Matrix\n")
        self.csvLog.text_widget.insert(END, "----------------------\n")
        self.csvLog.text_widget.insert(END, self.X)
        self.csvLog.text_widget.insert(END, "\n\n")
        self.csvLog.text_widget.insert(END, "Label Vector\n")
        self.csvLog.text_widget.insert(END, "----------------------\n")
        self.csvLog.text_widget.insert(END, self.y)
        self.csvLog.text_widget.see("end")
        self.csvLog.pack()


    def displayResult(self, dict):
        self.mainLog.text_widget.insert(END, "Done computing.\n")
        self.mainLog.text_widget.see("end")

        # Destroy if result window and log exists.
        try:
            self.window2.destroy()
            self.resultLog.destroy()
        except (NameError, AttributeError):
            pass

        # Create a new frame/window from root window to display CSV file
        self.window2 = Toplevel(self)
        self.window2.geometry("640x300")
        self.window2.title("Results of " + self.filename)

        # Adds a textbox
        # Height is the lines to show, width is the number of characters to show
        self.resultLog = Text2(self.window2, width=self.width, height=self.height)
        self.resultLog.text_widget.insert(END, "Training Set Size: " + str(dict["train_size"]) + "\n")
        self.resultLog.text_widget.insert(END, "Testing Set Size: " + str(dict["test_size"]) + "\n")
        self.resultLog.text_widget.insert(END, "Training Set Shape: " + str(dict["X_train.shape"]) + "\n")
        self.resultLog.text_widget.insert(END, "Testing Set Shape: " + str(dict["X_test.shape"]) + "\n")
        self.resultLog.text_widget.insert(END, str(dict["classifier"]) + "\n")
        self.resultLog.text_widget.insert(END, "Accuracy for " + self.algorithm + ": " + str(dict["accuracy"]) + "\n")
        self.resultLog.text_widget.insert(END, "Cross Validation for " + self.algorithm + ": " + str(dict["accuracy_list"].mean()) + "\n")
        self.resultLog.text_widget.insert(END, dict["report"] + "\n")
        self.resultLog.text_widget.see("end")
        self.resultLog.pack()

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


    def openFile(self):
        # Clear the widgets in the main frame every time "Open File" is clicked
        for widget in self.winfo_children():
            widget.destroy()

        # The full path of the file
        file = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("csv files","*.csv"),))

        if file:
            # Actual filename
            self.filename = os.path.basename(file)

            # Notify user that program is reading off the csv
            self.mainLog.text_widget.insert(END, "Reading '" + self.filename + "' from '" + file + "'.\n")
            self.mainLog.text_widget.see("end")

            # Dataframe created from the file
            self.df = pd.read_csv(file, sep=',')

            # Columns of dataframe
            cols = list(self.df.columns.values)

            # Create a listbox
            self.list_of_features = Listbox(self, selectmode=MULTIPLE, height=5, exportselection=0)
            self.list_of_features.bind('<<ListboxSelect>>', self.removeFeatures)
            self.list_of_label = Listbox(self, selectmode=SINGLE, height=5, exportselection=0)

            # Show a list of columns for user to check
            for column in cols:
                self.list_of_features.insert(END, column)
                self.list_of_label.insert(END, column)

            # Display label, listbox, and button
            Label(self, text="Select the feature columns", relief=RIDGE).pack()
            self.list_of_features.pack()
            Label(self, text="Select the label", relief=RIDGE).pack()
            self.list_of_label.pack()

            ok = Button(self, text="Okay", command=self.setUpMatrixes)
            ok.pack()


    def setUpMatrixes(self):
        # The selections of feature columns and labels
        columns = self.list_of_features.get(0, END)
        indexesForFeatureCols = self.list_of_features.curselection()
        selected_features = [columns[item] for item in indexesForFeatureCols]
        selected_label = self.list_of_label.get(ANCHOR)

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

        # Clear the widgets in the main frame
        for widget in self.winfo_children():
            widget.destroy()

        # Option Menu for choosing machine learning algorithms
        algorithms = ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine", "Multilayer Perceptron"]
        self.default = StringVar()
        self.default.set("Select an algorithm.")
        self.options = OptionMenu(self, self.default, *algorithms, command=self.selectedAlgorithm)
        self.options.pack()

        # Parameters frame
        self.parameterFrame = Frame(self, width=self.winfo_width()-200, height=self.winfo_height()-200)
        self.parameterFrame.pack()

        # Display the csv file
        self.displayFile()

    def selectedAlgorithm(self, algorithm):
        # Clear the widgets in the parameter frame when changing algorithm
        for widget in self.parameterFrame.winfo_children():
            widget.destroy()

        # Validation command
        # %d = Type of action (1=insert, 0=delete, -1 for others)
        # %P = value of the entry if the edit is allowed (all, focusin, focusout, forced)
        vcmdForInt = (self.parameterFrame.register(self.validateInt),
                      '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat = (self.parameterFrame.register(self.validateFloat),
                        '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForFloat2 = (self.parameterFrame.register(self.validateFloat2),
                         '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        vcmdForHiddenLayerSizes = (self.parameterFrame.register(self.validateHiddenLayerSizes),
                         '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        self.algorithm = algorithm

        # Load image
        load = Image.open(algorithm + ".png")

        # Load image
        load = load.resize((200, 200), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        # Labels can be text or images
        img = Label(self.parameterFrame, image=render)
        img.image = render
        img.pack()

        Label(self.parameterFrame, text="test_size", relief=RIDGE).pack()
        self.test_size = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForFloat)
        self.test_size.insert(0, 0.3)
        self.test_size.pack()

        Label(self.parameterFrame, text="random_state", relief=RIDGE).pack()
        self.random_state = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForInt)
        self.random_state.insert(0, 2)
        self.random_state.pack()

        Label(self.parameterFrame, text="cv", relief=RIDGE).pack()
        self.cv = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForInt)
        self.cv.insert(0, 10)
        self.cv.pack()


        if self.algorithm == "K-Nearest Neighbors":
            Label(self.parameterFrame, text="n_neighbors", relief=RIDGE).pack()
            self.n_neighbors = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForInt)
            self.n_neighbors.insert(0, 5)
            self.n_neighbors.pack()

        elif algorithm == "Random Forest":
            Label(self.parameterFrame, text="n_estimators", relief=RIDGE).pack()
            self.n_estimators = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForInt)
            self.n_estimators.insert(0, 19)
            self.n_estimators.pack()

        elif algorithm == "Multilayer Perceptron":
            Label(self.parameterFrame, text="max_iter", relief=RIDGE).pack()
            self.max_iter = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForInt)
            self.max_iter.insert(0, 100)
            self.max_iter.pack()

            Label(self.parameterFrame, text="alpha", relief=RIDGE).pack()
            self.alpha = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForFloat2)
            self.alpha.insert(0, 0.005)
            self.alpha.pack()

            Label(self.parameterFrame, text="hidden_layer_sizes", relief=RIDGE).pack()
            self.hidden_layer_sizes = Entry(self.parameterFrame, validate="all", validatecommand=vcmdForHiddenLayerSizes)
            self.hidden_layer_sizes.insert(0, 2)
            self.hidden_layer_sizes.pack()

        # Compute using the specified parameters
        submit = Button(self.parameterFrame, text="Submit", command=self.compute)
        submit.pack()

        # Notify user that program is reading off the csv
        self.mainLog.text_widget.insert(END, self.algorithm + " has been selected!\n")
        self.mainLog.text_widget.see("end")

    def validateInt(self, d, i, P, s, S, v, V, W):
        print("end", "OnValidate:\n")
        print("end", "d='%s'\n" % d)
        print("end", "i='%s'\n" % i)
        print("end", "P='%s'\n" % P)
        print("end", "s='%s'\n" % s)
        print("end", "S='%s'\n" % S)
        print("end", "v='%s'\n" % v)
        print("end", "V='%s'\n" % V)
        print("end", "W='%s'\n" % W)

        # Accept Integer values and empty string (for erasing the one only number)
        if P.isdigit() or P == "":
            return True
        else:
            self.mainLog.text_widget.insert(END, "Please enter an integer.\n")
            self.mainLog.text_widget.see("end")
            self.bell()
            return False

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
                self.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Float numbers are only allowed (ex: 0.3 or .3)!\n")
            self.mainLog.text_widget.see("end")
            self.bell()
            return False

    def validateFloat2(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            number = float(P)

            if (0.00001 <= number and number <= 1000.0):
                return True
            else:
                self.mainLog.text_widget.insert(END, "Float numbers must be between 0.00001 and 1000.0 (inclusive)!\n")
                self.mainLog.text_widget.see("end")
                self.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Float numbers are only allowed (ex: 0.00001 or .00001)!\n")
            self.mainLog.text_widget.see("end")
            self.bell()
            return False

    def validateHiddenLayerSizes(self, d, i, P, s, S, v, V, W):
        # Accept Float values and empty string (for erasing the one only number)
        if P == "":
            return True

        try:
            hidden_layer_sizes = P.split(",")

            if S.isdigit() or S == "," :
                return True
            else:
                self.mainLog.text_widget.insert(END, "Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
                self.mainLog.text_widget.see("end")
                self.bell()
                return False
        except ValueError:
            self.mainLog.text_widget.insert(END, "Hidden layer sizes should be separated by commas (ex: 2,3,4). This means there are 2 nodes in first hidden layer, 3 nodes in second hidden layer, and 4 nodes in the third hidden layer.!\n")
            self.mainLog.text_widget.see("end")
            self.bell()
            return False


    def compute(self):
        self.mainLog.text_widget.insert(END, "Computing...\n")
        self.mainLog.text_widget.see("end")

        # Split the dataframe dataset.
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(0.3), random_state=int(self.random_state.get()))

        classifier = None

        if self.algorithm == "K-Nearest Neighbors":
            # Instantiating KNN object
            classifier = KNeighborsClassifier(n_neighbors=int(self.n_neighbors.get()))
        elif self.algorithm == "Decision Tree":
            # Instantiating DecisionTreeClassifier object
            classifier = DecisionTreeClassifier()
        elif self.algorithm == "Random Forest":
            # Instantiating RandomForestClassifier object
            classifier = RandomForestClassifier(n_estimators=int(self.n_estimators.get()), bootstrap=True)
        elif self.algorithm == "Support Vector Machine":
            # LinearSVC classifier
            classifier = LinearSVC()
        elif self.algorithm == "Multilayer Perceptron":
            # instantiate model using:
            # a maximum of 1000 iterations (default = 200)
            # an alpha of 1e-5 (default = 0.001)
            # and a random state of 42 (for reproducibility)

            # Turn the string (containing commas) into a list
            modified_hidden_layer_sizes = self.hidden_layer_sizes.get().split(",")

            # Remove any empty string in the list
            modified_hidden_layer_sizes = [item.strip() for item in modified_hidden_layer_sizes if item.strip()]

            # Turn the list of strings into a tuple of int
            modified_hidden_layer_sizes = tuple([int(i) for i in modified_hidden_layer_sizes])

            classifier = MLPClassifier(max_iter=int(self.max_iter.get()), alpha=float(self.alpha.get()),
                                   hidden_layer_sizes=modified_hidden_layer_sizes)

        # fit the model with the training set
        classifier.fit(X_train, y_train)

        # Predict method is used for creating a prediction on testing data
        y_predict = classifier.predict(X_test)

        # Accuracy of testing data on predictive model
        accuracy = accuracy_score(y_test, y_predict)

        # Add #-fold Cross Validation with Supervised Learning
        accuracy_list = cross_val_score(classifier, self.X, self.y, cv=int(self.cv.get()), scoring='accuracy')

        # Report
        report = classification_report(y_test, y_predict, target_names=self.labels)

        # Dictionary containing information
        dict = {"train_size": 1.00 - float(self.test_size.get()), "test_size": float(self.test_size.get()), "X_train.shape": X_train.shape, "X_test.shape": X_test.shape, "classifier": classifier, "accuracy": accuracy, "accuracy_list": accuracy_list, "report": report}

        self.displayResult(dict)

    # Exit the client
    def client_exit(self):
        exit()

def main():
    # Creates the root window
    root = Tk()

    # Creates size of the window
    root.geometry("800x600")

    # Create an instance of window
    app = Window(root)

    # Show the window
    root.mainloop()


if __name__ == '__main__':
    main()
