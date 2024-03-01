import tkinter as tk
from tkinter import filedialog
import WorkingVersion1 as wv1


class GUI:

    def __init__(self, master):
        self.master = master
        master.title("Interactive Chat and File Locator")

        # Chat Window
        self.chat_frame = tk.Frame(master)
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_log = tk.Text(self.chat_frame)
        self.chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.chat_frame, command=self.chat_log.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_log.config(yscrollcommand=self.scrollbar.set)

        self.chat_input = tk.Entry(master)
        self.chat_input.pack(fill=tk.X)

        self.send_button = tk.Button(master, text="Send message", command=self.send_message)
        self.send_button.pack()

        self.url_label = tk.Label(master, text="Enter GIT URL or local path and select model afterwards:")
        self.url_label.pack()
        self.url_input = tk.Entry(master)
        self.url_input.pack(fill=tk.X)

        # File Locator
        self.file_button = tk.Button(master, text="Select model", command=self.locate_file)
        self.file_button.pack()

    # Query the LLM with inputted text and display the response
    def send_message(self):
        message = self.chat_input.get()
        self.chat_input.delete(0, tk.END)
        self.chat_log.insert(tk.END, f"Your question: {message}\n")
        self.chat_log.insert(tk.END, f"Response: {wv1.llm_reponse(message)}\n\n")

    # call setup method in WorkingVersion1.py with GIT path and model path
    def locate_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            wv1.setup(self.url_input.get(), file_path)
            #self.chat_log.insert(tk.END, f"Selected file: {file_path}\n")



def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()