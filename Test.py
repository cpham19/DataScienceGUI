from tkinter import *


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

# Create a Window Class
class Window:
    # Initialize PanedWindow widget
    def __init__(self):
        # Creates the root window
        self.root = Tk()

        # Creates size of the window
        self.root.geometry("800x600")

        # Set the title of our master widget
        self.root.title("Data Science GUI")

        # Width and height for window
        self.width = 800
        self.height = 640

        # Paned window for Top (main stuff) and Bottom (mainLog)
        main = PanedWindow(self.root, orient=VERTICAL, sashpad=1, sashrelief=RAISED)
        main.pack(fill=BOTH, expand=1)

        # Paned window for left (choosing features/label and parameters/algorithm) and right (displaying csv log and results log)
        top = PanedWindow(main, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)

        # Log for main stuff
        bottom = PanedWindow(main, orient=HORIZONTAL, sashpad=1, sashrelief=RAISED)

        main.add(top, height=440)
        main.add(bottom, height=200)

        # Paned Window for choosing features/label and parameters/algorithm
        left = Frame(top)

        # Paned window for CSV File and Results
        right = PanedWindow(top, orient=VERTICAL, sashpad=1, sashrelief=RAISED)

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

        right.add(self.labelFrameForCSVFile, height=220)
        right.add(self.labelFrameForResult, height=220)

        self.mainLog = Text2(bottom, width=self.width, height=self.height - 100)
        self.mainLog.text_widget.insert(END, "Started the Data Science GUI!\n")
        self.mainLog.text_widget.see("end")
        bottom.add(self.mainLog)

        self.root.mainloop()


# Create an instance of window
app = Window()

