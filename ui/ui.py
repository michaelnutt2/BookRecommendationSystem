"""
Class for the GUI that accepts in a user book and provides what the predicted rating will be
"""

from tkinter import *
from tkinter import messagebox
import model


class GUI:
    def __init__(self):
        # Set up the window
        self.window = Tk()
        self.window.title("Book Rating Prediction")
        self.window.config(padx=20, pady=20)
        self.window.resizable(False, False)

        self.model = model.load_model()

        # B
        self.submit = Button(text="Predict!", command=self.predict)
        self.user_entry = Entry()
        self.book_entry = Entry()
        user_label = Label(text="Enter User ID")
        book_label = Label(text="Enter Book ID")
        self.rating_label = Label(text="Predicted Rating: Unknown")

        user_label.grid(column=0, row=0)
        book_label.grid(column=0, row=1)
        self.user_entry.grid(column=1, row=0)
        self.book_entry.grid(column=1, row=1)
        self.submit.grid(column=0, row=2, columnspan=2)
        self.rating_label.grid(column=0, row=3, columnspan=2)

    def predict(self):
        if self.book_entry.get() is '' or self.user_entry.get() is '':
            return messagebox.showwarning('Warning', 'Empty Fields')

        value = model.new_prediction(self.user_entry, self.book_entry, self.model)

        rating = "Predicted Rating: " + str(value)

        self.rating_label.config(text=rating)

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    ui = GUI()
    ui.run()
