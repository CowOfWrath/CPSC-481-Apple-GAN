import os
import random
import tkinter.messagebox
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


def isReal():
    if int(rand[:-4]) < 1000:
        response = tkinter.messagebox.askokcancel("", "Congratulations!\nDo you want to play again?")
    else:
        response = tkinter.messagebox.askokcancel("", "Fooled you!\nDo you want to play again?")
    if response:
        pass
    else:
        destroy_window()


def isGenerated():
    if int(rand[:-4]) >= 1000:
        response = tkinter.messagebox.askokcancel("", "Congratulations!\nDo you want to play again?")
    else:
        response = tkinter.messagebox.askokcancel("", "Fooled you!\nDo you want to play again?")
    if response:
        pass
    else:
        destroy_window()


Button(window, text="REAL", width=10, command=isReal).grid(row=2, column=0, sticky=W)
Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=2, column=0, sticky=E)
Button(window, text="QUIT", width=10, command=destroy_window).grid(row=3, column=0, sticky=N)

window.mainloop()
