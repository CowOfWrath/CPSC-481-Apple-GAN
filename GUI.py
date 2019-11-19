import os
import random
import tkinter.messagebox
from tkinter import *


def restart():
    python = sys.executable
    os.execl(python, python, *sys.argv)


def destroy_window():
    window.destroy()
    exit()


def isReal():
    global rand, window, score, total
    if int(rand[:-4]) < 1000:
        response = tkinter.messagebox.askyesno("", "Congratulations!\nDo you want to play again?")
        score += 1
        total += 1
    else:
        response = tkinter.messagebox.askyesno("", "Fooled you!\nDo you want to play again?")
        total += 1
    if response:
        window.destroy()

        window = Tk()
        window.resizable(0, 0)
        window.title('Apple-GAN')
        question_string = "Is the image below real or AI-generated?\n"
        Label(window, text=question_string, font="none 12 bold").grid(row=0, column=0, sticky=W)
        # images from 001 to 947 are real, 1000 to 1998 are generated
        rand = random.choice(os.listdir(r"Grayscales"))
        print(rand)
        photo = PhotoImage(file="Grayscales\\" + rand)
        Label(window, image=photo).grid(row=1, column=0, sticky=N)
        score_string = "Current score: " + str(score) + " / " + str(total) + "\n"
        Label(window, text=score_string, font="none 12 bold").grid(row=2, column=0, sticky=W)

        Button(window, text="REAL", width=10, command=isReal).grid(row=3, column=0, sticky=W)
        Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=3, column=0, sticky=E)
        Button(window, text="QUIT", width=10, command=destroy_window).grid(row=4, column=0, sticky=N)

        window.mainloop()
    else:
        destroy_window()
        window.destroy()


def isGenerated():
    global rand, window, score, total
    if int(rand[:-4]) >= 1000:
        response = tkinter.messagebox.askyesno("", "Congratulations!\nDo you want to play again?")
        score += 1
        total += 1
    else:
        response = tkinter.messagebox.askyesno("", "Fooled you!\nDo you want to play again?")
        total += 1
    if response:
        window.destroy()

        window = Tk()
        window.resizable(0, 0)
        window.title('Apple-GAN')
        question_string = "Is the image below real or AI-generated?\n"
        Label(window, text=question_string, font="none 12 bold").grid(row=0, column=0, sticky=W)
        # images from 001 to 947 are real, 1000 to 1998 are generated
        rand = random.choice(os.listdir(r"Grayscales"))
        print(rand)
        photo = PhotoImage(file="Grayscales\\" + rand)
        Label(window, image=photo).grid(row=1, column=0, sticky=N)
        score_string = "Current score: " + str(score) + " / " + str(total) + "\n"
        Label(window, text=score_string, font="none 12 bold").grid(row=2, column=0, sticky=W)

        Button(window, text="REAL", width=10, command=isReal).grid(row=3, column=0, sticky=W)
        Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=3, column=0, sticky=E)
        Button(window, text="QUIT", width=10, command=destroy_window).grid(row=4, column=0, sticky=N)

        window.mainloop()
    else:
        destroy_window()
        window.destroy()


score = 0
total = 0

window = Tk()
window.resizable(0, 0)
window.title('Apple-GAN')

question_string = "Is the image below real or AI-generated?\n"
Label(window, text=question_string, font="none 12 bold").grid(row=0, column=0, sticky=W)

# images from 001 to 947 are real, 1000 to 1998 are generated
rand = random.choice(os.listdir(r"Grayscales"))
print(rand)
photo = PhotoImage(file="Grayscales\\" + rand)
Label(window, image=photo).grid(row=1, column=0, sticky=N)
score_string = "Current score: " + str(score) + " / " + str(total) + "\n"
Label(window, text=score_string, font="none 12 bold").grid(row=2, column=0, sticky=W)

Button(window, text="REAL", width=10, command=isReal).grid(row=3, column=0, sticky=W)
Button(window, text="GENERATED", width=10, command=isGenerated).grid(row=3, column=0, sticky=E)
Button(window, text="QUIT", width=10, command=destroy_window).grid(row=4, column=0, sticky=N)

window.mainloop()
