from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
import tkinter as tk
from tkinter import simpledialog, Button, Checkbutton, BooleanVar
from PIL import Image, ImageTk
import io
import threading
import time

# Initialize Tkinter root window
root = tk.Tk()
root.title("SD-Turbo Forever")
root.geometry("800x800")

# Load the image generation pipelines
pipe_text2img = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe_text2img.to("cuda")

pipe_img2img = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe_img2img.to("cuda")

# Flag to control the generation loop
generating = False
previous_image = None
use_img2img = BooleanVar()

# Function to start/stop the generation loop
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
            width = int(width_entry.get())
            height = int(height_entry.get())

            if use_img2img.get() and previous_image is not None:
                # Use img2img pipeline with the previous image
                init_image = previous_image.resize((width, height))
                generated_image = pipe_img2img(prompt=prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
            else:
                # Use text2img pipeline
                generated_image = pipe_text2img(prompt=prompt, width=width, height=height, num_inference_steps=1, guidance_scale=0.0).images[0]
                
            # Convert PIL image to Tkinter format
            image_bytes = io.BytesIO()
            generated_image.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            tk_image = ImageTk.PhotoImage(Image.open(image_bytes))
            
            # Update the image in the Tkinter label
            image_label.config(image=tk_image)
            image_label.image = tk_image

            # Store the generated image for future use if img2img is checked
            previous_image = generated_image

            # Dynamically update the Tkinter window size to fit the new image
            root.geometry(f"{width + 150}x{height + 250}")

        except Exception as e:
            pass
        # Pause to prevent overwhelming GPU resources
        time.sleep(2 if not generating else 0)

# Handle window close event
def on_close():
    global generating
    generating = False
    root.destroy()

# Create prompt input field and button
prompt_label = tk.Label(root, text="Enter prompt:", font=("Helvetica", 14))
prompt_label.pack(pady=10, anchor='n')

prompt_entry = tk.Entry(root, width=70, font=("Helvetica", 14))
prompt_entry.pack(pady=5, ipady=10)
prompt_entry.insert(0, "A cinematic shot of a baby racoon wearing an intricate italian priest robe.")

# Create width and height input fields on the same line
dimension_frame = tk.Frame(root)
dimension_frame.pack(pady=10)

width_label = tk.Label(dimension_frame, text="Enter width:", font=("Helvetica", 14))
width_label.grid(row=0, column=0, padx=5)

width_entry = tk.Entry(dimension_frame, width=10, font=("Helvetica", 14))
width_entry.grid(row=0, column=1, padx=5)
width_entry.insert(0, "512")

height_label = tk.Label(dimension_frame, text="Enter height:", font=("Helvetica", 14))
height_label.grid(row=0, column=2, padx=5)

height_entry = tk.Entry(dimension_frame, width=10, font=("Helvetica", 14))
height_entry.grid(row=0, column=3, padx=5)
height_entry.insert(0, "512")

# Create a checkbox for img2img
img2img_checkbox = Checkbutton(root, text="Use img2img", variable=use_img2img, font=("Helvetica", 14))
img2img_checkbox.pack(pady=10)

# Create Start/Stop button
generate_button = Button(root, text="Start", command=start_stop_generation, font=("Helvetica", 14))
generate_button.pack(pady=20)

# Label to display the generated image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Set the close protocol to ensure the program ends properly
root.protocol("WM_DELETE_WINDOW", on_close)

# Run the Tkinter main loop
root.mainloop() 