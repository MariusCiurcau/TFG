import threading
import time
import tkinter as tk
from tkinter import ttk

import imageio
from PIL import Image, ImageTk
from playsound import playsound

AUDIO = True

def loopSound():
    while True:
        playsound("Gegagedigedagedago.mp3", block=True)

def play_gif(root):
    gif_path = "Gegagedigedagedago.gif"  # Replace "your_gif.gif" with the path to your GIF file
    gif = imageio.get_reader(gif_path)
    frames = iter(gif)
    def update_frame():
        frame = next(frames)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    frame = next(frames)
    img = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=photo)
    label.pack()
    label.place(x=60, y=375, width=200, height=150)

    while True:
        try:
            update_frame()
            time.sleep(0.075)
        except StopIteration:
            frames = iter(gif)

class GUI:
    def __init__(self, master, images, texts):
        self.master = master
        self.master.title("Hip fracture detection")
        self.master.geometry("800x1050")

        # Title
        self.big_title_label = tk.Label(master, text='Hip fracture detection', font=("Helvetica", 20, "bold"))
        self.big_title_label.pack(pady=20)

        # Images
        self.image_frame = tk.Frame(master)
        self.image_frame.pack()

        self.images = [ImageTk.PhotoImage(image) for image in images.values()]

        for i, (title, img) in enumerate(images.items()):
            photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(self.image_frame, image=photo)
            image_label.image = photo  # Keep reference to avoid garbage collection
            image_label.grid(row=1, column=i, padx=5, pady=5)
            title_label = tk.Label(self.image_frame, text=title, font=("Helvetica", 14))
            title_label.grid(row=0, column=i, padx=5, pady=5)

        self.versions = list(list(texts.values())[0].keys())
        self.llms = list(texts.keys())

        default_version = self.versions[0]
        default_llm = self.llms[0]

        # Diagnosis
        self.diagnosis_label = tk.Label(master, text='Diagnosis', font=("Helvetica", 16, "bold"))
        self.diagnosis_label.pack(pady=20)

        # LLM selector
        self.llm_var = tk.StringVar(master)
        self.llm_var.set(default_llm)  # Default option
        self.llm_frame = tk.Frame(master)
        self.llm_frame.pack()

        self.text_label = tk.Label(self.llm_frame, text="Select LLM:")
        self.text_label.grid(row=0, column=0, padx=5, pady=5)

        for i, llm in enumerate(self.llms):
            radio_button = tk.Radiobutton(self.llm_frame, text=llm, variable=self.llm_var, value=llm,
                                          command=lambda: self.update_llm(texts, self.llm_var.get()))
            radio_button.grid(row=0, column=i + 1, padx=5, pady=5)

        # Radio buttons
        self.radio_var = tk.StringVar(master)
        self.radio_var.set(default_version) # Default option
        self.radio_frame = tk.Frame(master)
        self.radio_frame.pack()

        self.text_label = tk.Label(self.radio_frame, text="Select expertise level:")
        self.text_label.grid(row=0, column=0, padx=5, pady=5)

        for i, version in enumerate(self.versions):
            radio_button = tk.Radiobutton(self.radio_frame, text=version, variable=self.radio_var, value=version,
                                          command=lambda: self.update_text(texts[self.llm_var.get()][self.radio_var.get()]))
            radio_button.grid(row=0, column=i+1, padx=5, pady=5)

        """
        # Dropdown menu
        self.dropdown_frame = tk.Frame(master)
        self.dropdown_frame.pack()

        self.text_label = tk.Label(self.dropdown_frame, text="Select expertise level:")
        self.text_label.grid(row=0, column=0, padx=5, pady=5)

        self.dropdown_var = tk.StringVar(master)

        self.dropdown_var.set(self.options[0])  # Default option

        self.dropdown_menu = tk.OptionMenu(self.dropdown_frame, self.dropdown_var, *self.options,
                                           command=lambda selected_option: self.update_text(texts, selected_option))
        self.dropdown_menu.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        """

        # Textual explanation
        self.text_frame = tk.Frame(master)
        self.text_frame.pack(pady=10)

        """
        self.text_var = tk.StringVar(master)
        self.text_var.set(texts[self.options[0]])  # Default text
        self.text_label = tk.Label(self.text_frame, textvariable=self.text_var, font=("Helvetica", 14), wraplength=700, state='normal', justify='justify')
        self.text_label.pack()
        """

        self.text_widget = tk.Text(self.text_frame, font=("Helvetica", 14), wrap=tk.WORD, height=10,
                                   bg=master.cget('bg'), bd=0, width=60)
        self.text_widget.grid(row=1, column=0, columnspan=len(self.versions), padx=5, pady=5)
        self.text_widget.insert(tk.END, texts[default_llm][default_version])
        self.text_widget.config(state=tk.DISABLED)

        self.scrollbar = tk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.scrollbar.grid(row=1, column=len(self.versions), sticky="nsew")
        self.text_widget.config(yscrollcommand=self.scrollbar.set)

        self.llm_frame = tk.Frame(master)
        self.llm_frame.pack(pady=10)

        self.llm_text_var = tk.StringVar(master)
        self.llm_text_var.set('Text generated by {}'.format(default_llm))  # Default text
        self.text_label = tk.Label(self.llm_frame, textvariable=self.llm_text_var, font=("Helvetica", 12), state='normal', justify='left', fg='grey')
        self.text_label.pack()

        # Feedback
        self.diagnosis_label = tk.Label(master, text='Feedback', font=("Helvetica", 16, "bold"))
        self.diagnosis_label.pack(pady=20)

        # User input
        self.placeholder_text = "Enter your text here..."
        self.user_input = tk.Text(master, font=("Helvetica", 14), height=5, width=60, fg="gray")
        self.user_input.pack(pady=10)
        self.user_input.insert("1.0", self.placeholder_text)
        self.user_input.bind("<FocusIn>", self.remove_placeholder)
        self.user_input.bind("<FocusOut>", self.restore_placeholder)

        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_text)
        self.submit_button.pack(pady=5)

    def update_text(self, text):
        #self.text_var.set(texts[selected_option])
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, text)
        self.text_widget.config(state=tk.DISABLED)

    def update_llm(self, texts, selected_llm):
        self.update_text(texts[selected_llm][self.radio_var.get()])
        self.llm_text_var.set('Text generated by {}'.format(selected_llm))

    def submit_text(self):
        user_input_text = self.user_input.get("1.0", tk.END).strip()
        # Do something with the user input text, for example print it
        print("User input:", user_input_text)

    def remove_placeholder(self, event):
        if self.user_input.get("1.0", "end-1c") == self.placeholder_text:
            self.user_input.delete("1.0", "end-1c")
            self.user_input.config(fg="black")

    def restore_placeholder(self, event):
        if not self.user_input.get("1.0", "end-1c"):
            self.user_input.insert("1.0", self.placeholder_text)
            self.user_input.config(fg="gray")

def show_gui(images, texts):
    root = tk.Tk()
    if AUDIO:
        loopThread = threading.Thread(target=loopSound, name='backgroundMusicThread')
        loopThread.daemon = True  # shut down music thread when the rest of the program exits
        loopThread.start()
        gifThread = threading.Thread(target=play_gif, args=(root,), name='gifThread')
        gifThread.daemon = True  # shut down gif thread when the rest of the program exits
        gifThread.start()

    app = GUI(root, images, texts)
    root.mainloop()

#img1 = Image.open('../Datasets/Dataset/Femurs/resized_images/01.1_0.jpg')
#show_gui([img1], {'Básico': "Texto básico", 'Avanzado': "Texto avanzado"})
