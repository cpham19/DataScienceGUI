# Created using https://pythonprogramming.net/python-3-tkinter-basics-tutorial/ tutorial
#Import everything from Tkinter module
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from os import getcwd
import os as os
import csv
import pandas as pd

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
        file = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("csv files","*.csv"),))
        df = pd.read_csv(file, sep=',')
        print(df)

        if file:
            self.filename = os.path.basename(file)
            self.text = Label(self, text=self.filename)
            self.text.pack()

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
