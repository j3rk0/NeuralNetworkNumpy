import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
import numpy as np
from neural_network.neural_network import NeuralNetwork
from PIL import ImageGrab, Image, ImageFilter


class App:

    def convert_image(self, img):
        new_image = Image.new('L', (28, 28), 255)
        pic = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        new_image.paste(pic, (0, 0))
        # new_image.save("test.png")
        pixels = np.array(list(new_image.getdata())).reshape((1, 1, 28, 28))
        return ((255 - pixels) / 255) - .5

    def recognize_button_command(self):
        if self.net is None:
            # nessuna rete caricata
            return

        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab([x, y, x1, y1])
        img = self.convert_image(img)
        prediction = self.net.predict(img)
        self.result["text"] = f"result: {prediction[0]}"
        self.accuracy["text"] = "accuracy: " + "{:.2f}".format(np.max(prediction) * 100) + "%"
        return

    def __init__(self, root):
        self.net = None
        # setting title
        self.root = root
        root.title("MNIST recognizer")
        # setting window size
        width = 578
        height = 761
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        ft = tkfont.Font(family='Times', size=10)

        clear_btn = tk.Button(root, bg="#f0f0f0", font=ft, fg="#000000",
                              justify="center", text="clear",
                              command=self.clear_button_command)
        clear_btn.place(x=250, y=570, width=70, height=25)

        recognize_btn = tk.Button(root, bg="#f0f0f0", font=ft, fg="#000000",
                                  justify="center", text="recognize",
                                  command=self.recognize_button_command)
        recognize_btn.place(x=250, y=600, width=70, height=25)

        ##
        button_explore = tk.Button(top_win, bg="#f0f0f0", font=ft, fg="#000000",
                                   justify="center",
                                   text="load network",
                                   command=self.browse_files)
        button_explore.place(x=250, y=20, width=90, height=27)

        ##

        ft = tkfont.Font(family='Times', size=23)

        result_label = tk.Label(root, font=ft, fg="#333333",
                                justify="left", text="result: x")
        result_label.place(x=180, y=640, width=212, height=32)

        # metti label con testo "nessuna rete caricata"

        self.result = result_label

        ft = tkfont.Font(family='Times', size=18)

        accuracy_label = tk.Label(root, font=ft, fg="#333333",
                                  justify="left", text="accuracy: xx.xx")
        accuracy_label.place(x=180, y=680, width=216, height=30)

        self.accuracy = accuracy_label

        self.canvas = tk.Canvas(root, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.place(x=40, y=80, width=480, height=480)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.old_x = None
        self.old_y = None

    def browse_files(self):
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Select a File",
                                              filetypes=(("NeuralNetwork object",
                                                          "*.net"),
                                                         ("all files",
                                                          "*.")))
        net_name = filename.split("/").pop().split(".")[0]
        # setto eticetta a nome del file
        self.net = NeuralNetwork.from_file(filename)

    def clear_button_command(self):
        self.canvas.delete(tk.ALL)

    def reset(self, e):
        self.old_x = self.old_y = None

    def draw_lines(self, e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x,
                                    self.old_y,
                                    e.x,
                                    e.y,
                                    width=50,
                                    fill='black',
                                    capstyle=tk.ROUND,
                                    smooth=True)
        self.old_x = e.x
        self.old_y = e.y


if __name__ == "__main__":
    mnist_net = None  # NeuralNetwork().from_file("mnist_net.pkl")
    top_win = tk.Tk()
    app = App(top_win)  # window
    top_win.mainloop()
