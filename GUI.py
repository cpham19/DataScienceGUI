# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
#Import everything from Tkinter module
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from os import getcwd
import os as os
import csv
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
        fileMenu.add_command(label="Remove File", command=self.removeFile)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.client_exit)

        # Adds the options to the menu
        menu.add_cascade(label="File", menu=fileMenu)

        # Adds a textbox at the bottom
        # Height is the lines to show, width is the number of characters to show
        self.log = Text2(self.master, width=1280, height=90)
        self.log.text_widget.insert(END, "Started the Data Science GUI!\n")
        self.log.pack()

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

    def openFile(self):
        # The full path of the file
        file = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("csv files","*.csv"),))

        if file:
            self.filename = os.path.basename(file)

            self.log.text_widget.insert(END, "Reading '" + self.filename + "' from '" + file + ".\n")


            # Dataframe created from the file
            self.df = pd.read_csv(file, sep=',')

            # Insert
            self.log.text_widget.insert(END, str(self.df) + "\n")

    def removeFile(self):
        self.text.destroy()

    # Exit the client
    def client_exit(self):
        exit()

def main():
    # Creates the root window
    root = Tk()

    # Creates size of the window
    root.geometry("1280x720")

    # Create an instance of window
    app = Window(root)

    # Show the window
    root.mainloop()


if __name__ == '__main__':
    main()
