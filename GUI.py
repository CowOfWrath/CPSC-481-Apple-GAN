import os
import random
from tkinter import *

window = Tk()
window.resizable(0, 0)
window.title('Apple-GAN')

Label(window, text="Is the image below real or AI-generated?\n", font="none 12 bold").grid(row=0, column=0, sticky=W)

# images from 001 to 947 are real, 1000 to 1998 are generated
rand = random.choice(os.listdir(r"Grayscales"))
print(rand)
photo = PhotoImage(file="Grayscales\\" + rand)
Label(window, image=photo).grid(row=1, column=0, sticky=N)


def destroy_window():
    window.destroy()
    exit()


def retry():
    pass


def isReal():
    if int(rand[:-4]) < 1000:
        Label(window, text="Congratulations!\n", font="none 12 bold").grid(row=4, column=0, sticky=N)
    else:
        Label(window, text="Fooled you!\n", font="none 12 bold").grid(row=4, column=0, sticky=N)
    Button(window, text="RETRY", width=10, command=retry).grid(row=5, column=0, sticky=N)


def isGenerated():
    if int(rand[:-4]) >= 1000:
        Label(window, text="Congratulations!\n", font="none 12 bold").grid(row=3, column=0, sticky=N)
    else:
        Label(window, text="Fooled you!\n", font="none 12 bold").grid(row=4, column=0, sticky=N)
    Button(window, text="RETRY", width=10, command=retry).grid(row=5, column=0, sticky=N)


Button(window, text="REAL", width=10, command=isReal).grid(row=2, column=0, sticky=W)
Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=2, column=0, sticky=E)
Button(window, text="QUIT", width=10, command=destroy_window).grid(row=3, column=0, sticky=N)

window.mainloop()
