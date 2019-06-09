from tkinter import *

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
