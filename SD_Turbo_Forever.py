import tkinter as tk
from tkinter import simpledialog
from tkinter import BooleanVar
from PIL import Image, ImageTk
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import io
import threading
import time
import sys
import numpy as np
from tkinter.scrolledtext import ScrolledText

# Initialize Tkinter root window
root = tk.Tk()
root.title("SD-Turbo Forever")
root.configure(bg="#2b2b2b")  # Dark background manually set without ttkthemes
root.geometry("800x1024")  # Set narrower resolution to avoid resizing issues

# Label to display the generated image
image_frame = tk.Frame(root, width=1024, height=1024, bg="#2b2b2b")
image_frame.pack(pady=5)
image_label = tk.Label(image_frame, bg="#2b2b2b")
image_label.pack(expand=True)

# Redirect stdout and stderr to console_output widget
class ConsoleRedirect:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass  # No action needed for flushing



# Flag to control the generation loop
recording = False
video_frames = []
generating = False
previous_image = None
use_img2img = BooleanVar()
strength_var = tk.DoubleVar(value=0.5)

# Function to start/stop the recording loop
def start_stop_recording():
    global recording, video_frames
    if not recording:
        recording = True
        video_frames = []
        record_button.config(text="Stop Recording")
        size_select.config(state='disabled')
    else:
        recording = False
        record_button.config(text="Record Video")
        size_select.config(state='normal')
        create_video(video_frames)

def create_video(frames):
    # Create video from frames
    try:
        import cv2
        height, width = frames[0].size
        video = cv2.VideoWriter('generated_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (width, height))
        
        for frame in frames:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video.write(frame)
        
        video.release()
        print("Video saved as generated_video.avi\n")

    except ImportError:
        print("Please install OpenCV to create videos.\n")

def start_stop_generation():
    global generating
    if not generating:
        generating = True
        generate_button.config(text="Stop")
        threading.Thread(target=continuous_generate_image, daemon=True).start()
    else:
        generating = False
        generate_button.config(text="Start")

# Function to continuously generate and display images
def continuous_generate_image():
    global previous_image
    while generating:
        prompt = prompt_entry.get()
        try:
            width, height = map(int, size_var.get().split('x'))

            if use_img2img.get() and previous_image is not None:
                # Use img2img pipeline with the previous image
                init_image = previous_image.resize((width, height))
                num_inference_steps = max(1, int(2 / strength_var.get()))
                generated_image = pipe_img2img(prompt=prompt, image=init_image, num_inference_steps=num_inference_steps, strength=strength_var.get(), guidance_scale=0.0).images[0]
            else:
                # Use text2img pipeline
                generated_image = pipe_text2img(prompt=prompt, width=width, height=height, num_inference_steps=1, guidance_scale=0.0).images[0]
                
            # Convert PIL image to Tkinter format
            image_bytes = io.BytesIO()
            generated_image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            tk_image = ImageTk.PhotoImage(Image.open(image_bytes).resize((768, 768), Image.Resampling.LANCZOS))
            
            # Update the image in the Tkinter label
            image_label.config(image=tk_image)
            image_label.image = tk_image

            # Store the generated image for future use if img2img is checked
            previous_image = generated_image

            # Add frame to video if recording
            if recording:
                video_frames.append(generated_image)

        except Exception as e:
            print(f"Error: {str(e)}\n")
        # Pause to prevent overwhelming GPU resources
        time.sleep(2 if not generating else 0)

# Handle window close event
def on_close():
    global generating
    generating = False
    root.destroy()


# Load initial image into canvas
def load_initial_image():
    try:
        initial_image = Image.open('sd_turbo_forever.jpg')
        tk_image = ImageTk.PhotoImage(initial_image.resize((768, 768), Image.Resampling.LANCZOS))
        image_label.config(image=tk_image)
        image_label.image = tk_image
    except FileNotFoundError:
        print("Initial image not found. Please make sure 'SD-TURBO-FOREVER.jpg' is in the working directory.")

# Create prompt input field and button
prompt_label = tk.Label(root, text="Enter prompt:", bg="#2b2b2b", fg="#ffffff")
prompt_label.pack(pady=(0, 5), anchor='n')

prompt_entry = tk.Entry(root, bg="#000000", fg="white", insertbackground="white", width=root.winfo_width())
prompt_entry.pack(ipady=10, fill='x', padx=10)
prompt_entry.insert(0, "A cinematic shot of a baby racoon wearing an intricate Italian priest robe.")

# Create width and height input fields on the same line
dimension_frame = tk.Frame(root, bg="#2b2b2b")
dimension_frame.pack(pady=10, fill='x', padx=10)

size_label = tk.Label(dimension_frame, text="Select size:", bg="#2b2b2b", fg="#ffffff")
size_label.grid(row=0, column=0, padx=5)

size_options = ["128x128", "256x256", "384x384", "512x512", "640x640", "768x768", "896x896", "1024x1024"]
size_var = tk.StringVar(value="512x512")
size_select = tk.OptionMenu(dimension_frame, size_var, *size_options)
size_select.config(bg="#000000", fg="white")
size_select.grid(row=0, column=1, padx=5)

# Create a checkbox for img2img
img2img_checkbox = tk.Checkbutton(dimension_frame, text="Use img2img", variable=use_img2img, bg="#2b2b2b", fg="white", selectcolor="#3c3f41")
img2img_checkbox.grid(row=0, column=2, padx=5)

strength_label = tk.Label(dimension_frame, text="Strength:", bg="#2b2b2b", fg="#ffffff")
strength_label.grid(row=0, column=3, padx=5)
strength_slider = tk.Scale(dimension_frame, variable=strength_var, from_=0.1, to=1.0, resolution=0.1, orient='horizontal', bg="#2b2b2b", fg="white")
strength_slider.grid(row=0, column=4, padx=5)

# Create Record Video button
record_button = tk.Button(dimension_frame, text="Record Video", command=start_stop_recording, bg="#3c3f41", fg="white")
record_button.grid(row=0, column=6, padx=5)

# Create Start/Stop button
generate_button = tk.Button(dimension_frame, text="Start", command=start_stop_generation, bg="#3c3f41", fg="white")
generate_button.grid(row=0, column=5, padx=5)

# Label to display the generated image
image_frame = tk.Frame(root, width=1024, height=1024, bg="#2b2b2b")
image_frame.pack(pady=10)
image_label = tk.Label(image_frame, bg="#2b2b2b")
image_label.pack(expand=True)
load_initial_image()

# Create console output box
console_frame = tk.Frame(root, bg="#2b2b2b")
console_frame.pack(fill='both', expand=True, padx=5, pady=5)
console_output = ScrolledText(console_frame, wrap='word', bg="#000000", fg="white", insertbackground="white")
console_output.pack(fill='both', expand=True)

sys.stdout = ConsoleRedirect(console_output)
sys.stderr = ConsoleRedirect(console_output)


# Load the image generation pipelines
pipe_text2img = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe_text2img.to("cuda" if torch.cuda.is_available() else "cpu")

pipe_img2img = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe_img2img.to("cuda" if torch.cuda.is_available() else "cpu")

# Set the close protocol to ensure the program ends properly
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the Tkinter main loop
root.mainloop()
