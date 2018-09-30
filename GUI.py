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

        # # Creates a button instance
        # # Sets the quit button to the window
        # quitButton = Button(self, text="Quit", command=self.client_exit)
        # quitButton.place(x=0, y=0)

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
        # Create a new frame/window from root window
        self.window = Toplevel(self)
        self.window.geometry("640x480")
        self.window.title(self.filename)

        # Adds a textbox
        # Height is the lines to show, width is the number of characters to show
        self.sideLog = Text2(self.window, width=640, height=480)
        self.sideLog.text_widget.insert(END, self.df)
        self.sideLog.pack()

    def openFile(self):
        # The full path of the file
        file = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("csv files","*.csv"),))

        if file:
            # Actual filename
            self.filename = os.path.basename(file)

            # Notify user that program is reading off the csv
            self.mainLog.text_widget.insert(END, "Reading '" + self.filename + "' from '" + file + "'.\n")

            # Dataframe created from the file
            self.df = pd.read_csv(file, sep=',')

            # Display the csv file
            self.displayFile()

            # Option Menu for choosing machine learning algorithms
            algorithms = ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine", "Multilayer Perceptron"]
            self.default = StringVar()
            self.default.set("Select an algorithm.")
            self.options = OptionMenu(self, self.default, *algorithms, command=self.selectedAlgorithm)
            self.options.pack()

            # Parameters frame
            self.parameterFrame = Frame(self, width=self.winfo_width()-200, height=self.winfo_height()-200)
            self.parameterFrame.pack()


    def selectedAlgorithm(self, algorithm):
        if algorithm == "K-Nearest Neighbors":
            Label(self.parameterFrame, text="n_neighbors").grid(row=0, column=0)
            self.n_neighbors = Entry(self.parameterFrame)
            self.n_neighbors.grid(row=0, column=1)
            self.n_neighbors.insert(0, 5)

            Label(self.parameterFrame, text="weights").grid(row=0, column=2)
            self.weights = Entry(self.parameterFrame)
            self.weights.grid(row=0, column=3)
            self.weights.insert(0, "uniform")

            Label(self.parameterFrame, text="algorithm").grid(row=0, column=4)
            self.algorithm = Entry(self.parameterFrame)
            self.algorithm.grid(row=0, column=5)
            self.algorithm.insert(0, "auto")

            Label(self.parameterFrame, text="leaf_size").grid(row=1, column=0)
            self.leaf_size = Entry(self.parameterFrame)
            self.leaf_size.grid(row=1, column=1)
            self.leaf_size.insert(0, 30)

            Label(self.parameterFrame, text="p").grid(row=1, column=2)
            self.p = Entry(self.parameterFrame)
            self.p.grid(row=1, column=3)
            self.p.insert(0, 2)

            Label(self.parameterFrame, text="metric").grid(row=1, column=4)
            self.metric = Entry(self.parameterFrame)
            self.metric.grid(row=1, column=5)
            self.metric.insert(0, "minkowsi")

            Label(self.parameterFrame, text="metric_params").grid(row=2, column=0)
            self.metric_params = Entry(self.parameterFrame)
            self.metric_params.grid(row=2, column=1)
            self.metric_params.insert(0, "None")

            Label(self.parameterFrame, text="n_jobs").grid(row=2, column=2)
            self.n_jobs = Entry(self.parameterFrame)
            self.n_jobs.grid(row=2, column=3)
            self.n_jobs.insert(0, "None")

            submit = Button(self.parameterFrame, text="Submit", command=self.knn)
            submit.grid(row=5, column=3)

    def knn(self):
        # Feature columns
        feature_cols = list(self.df.columns.values)
        feature_cols.remove("species")

        # Filter the dataframe to show data from these feature columns
        X = self.df[feature_cols]

        # Create a label vector on label 'species'
        y = self.df['species']

        # Split the dataframe dataset. 70% of the data is training data and 30% is testing data using random_state 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

        # Instantiating KNN object
        knn = KNeighborsClassifier(n_neighbors=int(self.n_neighbors.get()))

        # Fit method is used for creating a trained model on the training sets for KNN classifier
        knn.fit(X_train, y_train)

        # Predict method is used for creating a predictive model using testing set
        y_predict_knn = knn.predict(X_test)

        # Accuracy of testing data on predictive model
        knn_accuracy = accuracy_score(y_test, y_predict_knn)

        # Print accuracy
        print("Accuracy for KNeighbors Classifier: " + str(knn_accuracy))

        # Instantiating DecisionTreeClassifier object
        my_DecisionTree = DecisionTreeClassifier(random_state=2)

        # Fit method is used for creating a trained model on the training sets for decisiontreeclassifier
        my_DecisionTree.fit(X_train, y_train)

        # Predict method is used for creating a predictive model using testing set
        y_predict_dt = my_DecisionTree.predict(X_test)

        # Accuracy of testing data on predictive model
        dt_accuracy = accuracy_score(y_test, y_predict_dt)

        # Print accuracy
        print("Accuracy for Decision Tree Classifier: " + str(dt_accuracy))


    # Exit the client
    def client_exit(self):
        exit()

def main():
    # Creates the root window
    root = Tk()

    # Creates size of the window
    root.geometry("640x480")

    # Create an instance of window
    app = Window(root)

    # Show the window
    root.mainloop()


if __name__ == '__main__':
    main()
