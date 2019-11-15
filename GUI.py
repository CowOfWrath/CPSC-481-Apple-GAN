from tkinter import *

window = Tk()
window.resizable(0, 0)
window.title('Apple-GAN')

Label(window, text="Is the image below real or AI-generated?\n", font="none 12 bold").grid(row=0, column=0, sticky=W)

photo = PhotoImage(file="Grayscales\\001.gif")
Label(window, image=photo).grid(row=1, column=0, sticky=N)


def destroy_window():
    window.destroy()
    exit()


def isReal():
    pass


def isGenerated():
    pass


Button(window, text="REAL", width=10, command=isReal).grid(row=2, column=0, sticky=W)
Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=2, column=0, sticky=E)
Button(window, text="QUIT", width=10, command=destroy_window).grid(row=3, column=0, sticky=N)

window.mainloop()
