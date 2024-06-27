import tkinter as tk
from tkinter import messagebox
from picamera import PiCamera
from time import sleep
from gpiozero import LED

led = LED(17)

def start_recording():
    try:
        duration = int(duration_entry.get())
        if duration <= 0:
            messagebox.showerror("Error", "Duration must be greater than 0")
            return
        camera = PiCamera()
        camera.start_recording('/home/research/Desktop/ZebraFishTracker/videos/fish.h264')
        led.on()
        sleep(duration)
        camera.stop_recording()
        camera.stop_preview()
        camera.close()
        led.off()
        messagebox.showinfo("Info", f"Recording finished. Video saved to /home/research/Desktop/ZebraFishTracker/videos/fish.h264")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for duration")

# Create the main window
root = tk.Tk()
root.title("Zebra Fish Camera Recorder")

# Create and place the label and entry for duration
duration_label = tk.Label(root, text="Enter duration (seconds):")
duration_label.pack(pady=5)

duration_entry = tk.Entry(root)
duration_entry.pack(pady=5)

# Create and place the record button
record_button = tk.Button(root, text="Start Recording", command=start_recording)
record_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
