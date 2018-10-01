# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
#Import everything from Tkinter module
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from os import getcwd
import os as os
import csv

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

# Create a Text widget that uses pixels for dimensions (width and height)
# Taken from http://code.activestate.com/recipes/578887-text-widget-width-and-height-in-pixels-tkinter/
class Text2(Frame):
    def __init__(self, master, width=0, height=0, **kwargs):
        self.width = width
        self.height = height

        Frame.__init__(self, master, width=self.width, height=self.height)
        self.text_widget = Text(self, **kwargs)
        self.text_widget.pack(expand=YES, fill=BOTH)

    def pack(self, *args, **kwargs):
        Frame.pack(self, *args, **kwargs)
        self.pack_propagate(False)

    def grid(self, *args, **kwargs):
        Frame.grid(self, *args, **kwargs)
        self.grid_propagate(False)


# Create a class that inherits from Frame class
class Window(Frame):

    # Initialize master widget
    def __init__(self, master=None):

        # Parameters sent to Frame class
        Frame.__init__(self, master)

        # Reference to master widget (tk window)
        self.master = master

        # Run
        self.init_window()


    # Initialize window
    def init_window(self):
        # Set the title of our master widget
        self.master.title("Data Science GUI")

        # Allows the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

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

        # Adds a textbox at the bottom
        # Height is the lines to show, width is the number of characters to show
        self.mainLog = Text2(self.master, width=640, height=100)
        self.mainLog.text_widget.insert(END, "Started the Data Science GUI!\n")
        self.mainLog.pack()

    # def showImage(self):
    #     load = Image.open("chat.png")
    #     render = ImageTk.PhotoImage(load)
    #
    #     # labels can be text or images
    #     img = Label(self, image=render)
    #     img.image = render
    #     img.place(x=0, y=0)


    # Display the csv file in text
    def displayFile(self):
        # Create a new frame/window from root window to display CSV file
        self.window = Toplevel(self)
        self.window.geometry("640x300")
        self.window.title(self.filename)

        # Adds a textbox
        # Height is the lines to show, width is the number of characters to show
        self.sideLog = Text2(self.window, width=640, height=300)
        self.sideLog.text_widget.insert(END, "Feature Matrix\n")
        self.sideLog.text_widget.insert(END, "----------------------\n")
        self.sideLog.text_widget.insert(END, self.X)
        self.sideLog.text_widget.insert(END, "\n\n")
        self.sideLog.text_widget.insert(END, "Label Vector\n")
        self.sideLog.text_widget.insert(END, "----------------------\n")
        self.sideLog.text_widget.insert(END, self.y)
        self.sideLog.pack()

    def displayResult(self):
        # Destroy if result window and log exists.
        try:
            self.window2.destroy()
            self.sideLog2.destroy()
        except (NameError, AttributeError):
            pass

        # Create a new frame/window from root window to display CSV file
        self.window2 = Toplevel(self)
        self.window2.geometry("640x300")
        self.window2.title("Results of " + self.filename)

        # Adds a textbox
        # Height is the lines to show, width is the number of characters to show
        self.sideLog2 = Text2(self.window2, width=640, height=300)
        self.sideLog2.text_widget.insert(END, "Accuracy for " + self.algorithm + ": " + str(self.accuracy) + "\n")
        self.sideLog2.text_widget.insert(END, self.report)
        self.sideLog2.pack()


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

            # Dataframe created from the file
            self.df = pd.read_csv(file, sep=',')

            # Columns of dataframe
            cols = list(self.df.columns.values)

            # Create a listbox
            self.list_of_columns = Listbox(self, selectmode=MULTIPLE, height=5, exportselection=0)
            self.list_of_label = Listbox(self, selectmode=SINGLE, height=5, exportselection=0)

            # Show a list of columns for user to check
            for column in cols:
                self.list_of_columns.insert(END, column)
                self.list_of_label.insert(END, column)

            # Display label, listbox, and button
            Label(self, text="Select the feature columns").pack()
            self.list_of_columns.pack()
            Label(self, text="Select the label").pack()
            self.list_of_label.pack()

            ok = Button(self, text="Okay", command=self.setUpMatrixes)
            ok.pack()


    def setUpMatrixes(self):
        # The selections of feature columns and labels
        columns = self.list_of_columns.get(0, END)
        indexesForFeatureCols = self.list_of_columns.curselection()
        selected_columns = [columns[item] for item in indexesForFeatureCols]
        indexForLabel = self.list_of_label.curselection()
        selected_label = [columns[item] for item in indexForLabel]

        # Feature matrix and label vector
        self.X = self.df[selected_columns]
        self.y = self.df[selected_label[0]]

        # Labels
        self.labels = self.df[selected_label[0]].unique()

        # Number of features and labels
        self.numberOfFeatures = len(selected_columns)
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

        Label(self.parameterFrame, text="test_size").grid(row=0, column=0)
        self.test_size = Entry(self.parameterFrame)
        self.test_size.grid(row=0, column=1)
        self.test_size.insert(0, 0.3)

        Label(self.parameterFrame, text="random_state").grid(row=0, column=2)
        self.random_state = Entry(self.parameterFrame)
        self.random_state.grid(row=0, column=3)
        self.random_state.insert(0, 2)

        self.algorithm = algorithm

        if self.algorithm == "K-Nearest Neighbors":
            Label(self.parameterFrame, text="n_neighbors").grid(row=0, column=4)
            self.n_neighbors = Entry(self.parameterFrame)
            self.n_neighbors.grid(row=0, column=5)
            self.n_neighbors.insert(0, 5)

        elif algorithm == "Decision Tree":
            print("PLACEHOLDER")

        elif algorithm == "Random Forest":
            Label(self.parameterFrame, text="n_estimators").grid(row=0, column=4)
            self.n_estimators = Entry(self.parameterFrame)
            self.n_estimators.grid(row=0, column=5)
            self.n_estimators.insert(0, 19)

        elif algorithm == "Multilayer Perceptron":
            Label(self.parameterFrame, text="max_iter").grid(row=0, column=4)
            self.max_iter = Entry(self.parameterFrame)
            self.max_iter.grid(row=0, column=5)
            self.max_iter.insert(0, 100)

            Label(self.parameterFrame, text="alpha").grid(row=1, column=0)
            self.alpha = Entry(self.parameterFrame)
            self.alpha.grid(row=1, column=1)
            self.alpha.insert(0, 0.005)

            Label(self.parameterFrame, text="hidden_layer_sizes").grid(row=1, column=2)
            self.hidden_layer_sizes = Entry(self.parameterFrame)
            self.hidden_layer_sizes.grid(row=1, column=3)
            self.hidden_layer_sizes.insert(0, 2)

        submit = Button(self.parameterFrame, text="Submit", command=self.compute)
        submit.grid(row=3, column=3)


    def compute(self):
        # Split the dataframe dataset. 70% of the data is training data and 30% is testing data using random_state 2
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(self.test_size.get()), random_state=int(self.random_state.get()))

        if self.algorithm == "K-Nearest Neighbors":
            # Instantiating KNN object
            knn = KNeighborsClassifier(n_neighbors=int(self.n_neighbors.get()))

            # Fit method is used for creating a trained model on the training sets for KNN classifier
            knn.fit(X_train, y_train)

            # Predict method is used for creating a predictive model using testing set
            y_predict = knn.predict(X_test)


        elif self.algorithm == "Decision Tree":
            # Instantiating DecisionTreeClassifier object
            my_DecisionTree = DecisionTreeClassifier()

            # Fit method is used for creating a trained model on the training sets for decisiontreeclassifier
            my_DecisionTree.fit(X_train, y_train)

            # Predict method is used for creating a predictive model using testing set
            y_predict = my_DecisionTree.predict(X_test)

        elif self.algorithm == "Random Forest":
            # Instantiating RandomForestClassifier object
            my_RandomForest = RandomForestClassifier(n_estimators=int(self.n_estimators.get()), bootstrap=True)

            # Fit method is used for creating a trained model on the training sets for RandomForestClassifier
            my_RandomForest.fit(X_train, y_train)

            # Predict method is used for creating a prediction on testing data
            y_predict = my_RandomForest.predict(X_test)


        elif self.algorithm == "Multilayer Perceptron":
            # instantiate model using:
            # a maximum of 1000 iterations (default = 200)
            # an alpha of 1e-5 (default = 0.001)
            # and a random state of 42 (for reproducibility)
            my_MLP = MLPClassifier(max_iter=int(self.max_iter.get()), alpha=float(self.alpha.get()),
                                   hidden_layer_sizes=((int((self.numberOfFeatures + self.numberOfLabels) / 2)),),
                                   random_state=42)

            # fit the model with the training set
            my_MLP.fit(X_train, y_train)

            # Predict method is used for creating a prediction on testing data
            y_predict = my_MLP.predict(X_test)


        # Accuracy of testing data on predictive model
        self.accuracy = accuracy_score(y_test, y_predict)

        self.report = classification_report(y_test, y_predict, target_names=self.labels)

        self.displayResult()

    # Exit the client
    def client_exit(self):
        exit()

def main():
    # Creates the root window
    root = Tk()

    # Creates size of the window
    root.geometry("640x300")

    # Create an instance of window
    app = Window(root)

    # Show the window
    root.mainloop()


if __name__ == '__main__':
    main()
