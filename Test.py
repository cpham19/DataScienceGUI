from tkinter import *


m1 = PanedWindow()
m1.pack(fill=BOTH, expand=1)

m2 = PanedWindow(m1, orient=VERTICAL)
m1.add(m2)

top = PanedWindow(m1, orient=HORIZONTAL)
m2.add(top)

left = PanedWindow(top, orient=VERTICAL)
label1 = Label(top, text="left pane of left")
left.add(label1)
top.add(left)

right = PanedWindow(top, orient=VERTICAL)
label3 = Label(top, text="top pane of right")
label4 = Label(top, text="bottom pane of right")
right.add(label3)
right.add(label4)
top.add(right)


bottom = Label(m2, text="bottom pane")
m2.add(bottom)

mainloop()