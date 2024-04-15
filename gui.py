import tkinter as tk
import webbrowser
import subprocess
root = tk.Tk()
root.title('G-Control')

canvas = tk.Canvas(root, bg='#0f1116', width=12800, height=720)
canvas.pack()

frame = tk.Frame(root, bg='#0f1116')
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.25, anchor='n')

html_button = tk.Button(frame, text='ABOUT US', font=('Arial', 14), bg='#3e2093', fg='white', command=lambda: webbrowser.open_new_tab('file:///home/home/Desktop/Project_G_control/index.html'))
html_button.pack(side='left', padx=10, pady=10)

script_button = tk.Button(frame, text='START', font=('Arial', 14), bg='#3e2093', fg='white', command=lambda: subprocess.call(['python3', 'g.py']))
script_button.pack(side='right', padx=20, pady=20)

label = tk.Label(root, text=' Welcome to Gesture Control!', font=('Arial', 25), fg='white', bg='#0f1116')
label.place(relx=0.5, rely=0.1, anchor='n')

image = tk.PhotoImage(file='pic.png')
image_label = tk.Label(root, bg='#0f1116', image=image)
image_label.place(relx=0.5, rely=0.2, anchor='n')

root.mainloop()

