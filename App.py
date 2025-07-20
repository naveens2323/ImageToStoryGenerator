import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Frame, Label, Button, Text, Listbox, Spinbox , ACTIVE , ttk
import time
import pyttsx3
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator 
import cv2
import random
import threading
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import wave
import pygame
from requests.exceptions import ChunkedEncodingError




# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

translator = GoogleTranslator(source="en", target="kn")
tts_engine = pyttsx3.init()

# Dictionary to store efficiency metrics
efficiency_metrics = {
    "image_processing_time": 0,
    "story_generation_time": 0,
    "translation_time": 0,
    "audio_generation_time": 0
}

def update_efficiency_metrics():
    """
    Updates efficiency metrics on the GUI.
    """
    metrics_text = (
        f"📸 Image Processing: {efficiency_metrics['image_processing_time']:.2f} sec\n"
        f"📖 Story Generation: {efficiency_metrics['story_generation_time']:.2f} sec\n"
        f"🌍 Translation: {efficiency_metrics['translation_time']:.2f} sec\n"
        f"🔊 Audio Generation: {efficiency_metrics['audio_generation_time']:.2f} sec"
    )
    efficiency_label.config(text=metrics_text)

SAVE_DIR = "saved_audio"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

stories = []
translator = GoogleTranslator(source="auto", target="kn")
engine = pyttsx3.init()
# Define paths for saving stories and audio
STORY_SAVE_PATH = "generated_stories"
AUDIO_SAVE_PATH = "generated_audio"

# Ensure directories exist
os.makedirs(STORY_SAVE_PATH, exist_ok=True)
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

LOCAL_IMAGE_FOLDER = "C:\\Users\\ansl6\\Downloads\\NAVEEN DS PROJECTS\\IR FINAL\\IMAGES"

def speak_kannada(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="kn")
    tts.save(filename)
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

def load_image(path):
    return Image.open(path).convert("RGB")

def generate_caption(image):
    start_time = time.time()
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
    efficiency_metrics["image_processing_time"] = time.time() - start_time
    update_efficiency_metrics()
    return blip_processor.decode(caption_ids[0], skip_special_tokens=True)

story_prompts = [
    "An unexpected journey begins when...", "A mysterious event changes everything...",
    "A hero rises in the face of danger...", "A magical world unfolds before them...",
    "A secret from the past resurfaces...", "A race against time begins...",
    "A lost artifact holds the key to...", "A twist of fate leads them to...",
    "A battle between good and evil ensues...", "An ancient prophecy reveals the truth..."
]

def generate_stories():
    start_time = time.time()

    global stories  
    caption = caption_label.cget("text").replace("Caption: ", "").strip()
    
    if not caption:
        messagebox.showerror("Error", "No caption generated. Upload or capture an image first.")
        return

    count = int(story_count_spinbox.get())
    stories.clear()  # Clear previous stories before generating new ones

    def generate_story_thread():
        new_stories = []  # Temporary list to hold new stories
        for _ in range(count):
            prompt = f"{random.choice(story_prompts)} {caption}"
            inputs = gpt_tokenizer.encode_plus(prompt, return_tensors="pt", max_length=100, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = gpt_model.generate(
                    inputs["input_ids"], 
                    max_length=200,  
                    temperature=0.9,  
                    top_k=50,  
                    top_p=0.95,  
                    repetition_penalty=1.3
                )

            story = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if story not in new_stories:  
                new_stories.append(story)  

        stories.extend(new_stories)  # Update global stories list
        window.after(0, update_story_listbox, new_stories)  # Sync with UI

        efficiency_metrics["story_generation_time"] = time.time() - start_time
        update_efficiency_metrics()

    threading.Thread(target=generate_story_thread, daemon=True).start()


def update_story_listbox(stories):
    story_listbox.delete(0, tk.END)
    for i, story in enumerate(stories):
        story_listbox.insert(tk.END, f"Story {i + 1}")
    
    def on_story_select(event):
        selected_index = story_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            story_text.delete("1.0", tk.END)
            story_text.insert(tk.END, stories[index])
            kannada_text.delete("1.0", tk.END)
            kannada_text.insert(tk.END, translator.translate(stories[index]))
    
    story_listbox.bind("<<ListboxSelect>>", on_story_select)



def capture_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows

    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access the camera.")
        return

    time.sleep(2)  # Allow camera to adjust exposure

    for _ in range(10):  # Try capturing multiple times
        ret, frame = cap.read()
        if ret and np.mean(frame) > 10:  # Ensure image is not black
            break
        time.sleep(0.1)  # Small delay before retrying

    cap.release()  # Release the webcam

    if not ret or np.mean(frame) <= 10:
        messagebox.showerror("Error", "Failed to capture a clear image.")
        return

    img_path = "captured_image.jpg"
    cv2.imwrite(img_path, frame)  # Save image

    # Display image in UI
    img = Image.open(img_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # Process the image
    caption = generate_caption(load_image(img_path))
    caption_label.config(text=f"Caption: {caption}")
    stories = generate_stories(caption)

    save_story(stories)

    story_listbox.delete(0, tk.END)
    for i in range(10):
        story_listbox.insert(tk.END, f"Story {i + 1}")



    def on_story_select(event):
        selected_index = story_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            story = stories[index]

            start_time = time.time()
            kannada_story = translator.translate(story)
            efficiency_metrics["translation_time"] = time.time() - start_time
            update_efficiency_metrics()

            story_text.delete("1.0", tk.END)
            story_text.insert(tk.END, story)

            kannada_text.delete("1.0", tk.END)
            kannada_text.insert(tk.END, kannada_story)

    story_listbox.bind("<<ListboxSelect>>", on_story_select)

# Upload image
def upload_image(source):
    file_types = [("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff;*.avif")]

    if source == "device":
        file_path = filedialog.askopenfilename(title="Select an Image from Device", filetypes=file_types)
    else:
        file_path = filedialog.askopenfilename(initialdir=LOCAL_IMAGE_FOLDER, title="Select an Image from Project Folder", filetypes=file_types)

    if file_path:
        process_image(file_path)

def process_image(img_path):
    img = Image.open(img_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    
    start_time = time.time()
    caption_label.config(text=f"Caption: {generate_caption(load_image(img_path))}")
    efficiency_metrics["translation_time"] = time.time() - start_time
    update_efficiency_metrics()

def read_english_story():
    story = story_text.get("1.0", tk.END).strip()
    if story:
        start_time = time.time()
        tts_engine.say(story)
        tts_engine.runAndWait()
        efficiency_metrics["audio_generation_time"] = time.time() - start_time
        update_efficiency_metrics()


def convert_text_to_speech(story_in_kannada, filename):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            tts = gTTS(text=story_in_kannada, lang="kn")
            tts.save(filename)
            return True  # Success
        except ChunkedEncodingError:
            retries += 1
            print(f"Retrying... Attempt {retries}/{max_retries}")
            time.sleep(2)  # Wait for 2 seconds before retrying
    print("Failed to convert text to speech after several attempts.")
    return False  # Failure


def read_kannada_story(kannada_text):
    story_in_kannada = kannada_text.get("1.0", tk.END).strip()
    
    if story_in_kannada:
        start_time = time.time()

        # Get the current number of files in the directory and determine the next file number
        existing_files = os.listdir(SAVE_DIR)
        max_number = 0
        for file in existing_files:
            if file.endswith(".mp3"):
                # Extract the number from filenames like 1.mp3, 2.mp3, etc.
                try:
                    number = int(file.split('.')[0])
                    max_number = max(max_number, number)
                except ValueError:
                    pass
        
        # Increment the number for the new file
        file_number = max_number + 1
        filename = os.path.join(SAVE_DIR, f"{file_number}.mp3")

        # Convert text to speech and save
        try:
            tts = gTTS(text=story_in_kannada, lang="kn")
            tts.save(filename)  # Save the file with the ordered name
            print(f"File saved as {filename}")
        except Exception as e:
            print(f"Error during conversion: {e}")
            return
        
        # Play the saved audio file using pygame.mixer
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            # Wait until the audio finishes playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error during playback: {e}")

        # Update the efficiency metrics
        efficiency_metrics["audio_generation_time"] = time.time() - start_time
        update_efficiency_metrics()

def save_audio(language):
    selected_story = story_listbox.get(ACTIVE)  # Get the selected story
    if not selected_story:
        messagebox.showerror("Error", "No story selected.")
        return
    messagebox.showinfo("Saving Audio", f"Saving audio in {language}...")  # Replace with actual saving logic

def save_story(language):
    selected_story = story_listbox.get(ACTIVE)  # Get the selected story
    if not selected_story:
        messagebox.showerror("Error", "No story selected.")
        return
    messagebox.showinfo("Saving Story", f"Saving story in {language}...")  # Replace with actual saving logic


def estimate_recording_duration(text):
    """Estimate duration based on story length (assuming 150 words per minute)."""
    words_per_minute = 150
    word_count = len(text.split())
    return word_count / words_per_minute * 60  # Convert minutes to seconds

def text_to_speech(text, filename):
    """Converts text to speech and saves as an audio file."""
    engine.save_to_file(text, filename)
    engine.runAndWait()

def text_to_speech(text, filename, lang="en"):
    """Convert text to speech and save as an audio file."""
    tts = gTTS(text=text, lang=lang)
    mp3_filename = filename.replace(".wav", ".mp3")  # gTTS only supports MP3
    tts.save(mp3_filename)
    
    # Convert MP3 to WAV if needed
    if filename.endswith(".wav"):
        audio = AudioSegment.from_mp3(mp3_filename)
        audio.export(filename, format="wav")

def record_audio(language):
    """Records and saves English, Kannada, or both story audios."""
    selected_index = story_listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No story selected.")
        return

    selected_index = selected_index[0]
    selected_story = stories[selected_index]

    if language in ["english", "both"]:
        english_filename = os.path.join(AUDIO_SAVE_PATH, f"story_english_{selected_index}.wav")
        text_to_speech(selected_story, english_filename, lang="en")
        messagebox.showinfo("Success", f"English audio saved:\n{english_filename}")

    if language in ["kannada", "both"]:
        translator = GoogleTranslator(source="auto", target="kn")
        kannada_story = translator.translate(selected_story)
        
        if not kannada_story.strip():
            messagebox.showerror("Error", "Kannada translation failed.")
            return
        
        kannada_filename = os.path.join(AUDIO_SAVE_PATH, f"story_kannada_{selected_index}.wav")
        text_to_speech(kannada_story, kannada_filename, lang="kn")
        messagebox.showinfo("Success", f"Kannada audio saved:\n{kannada_filename}")

def save_story(language):
    """Saves the selected story in English, Kannada, or both."""
    global stories  

    selected_index = story_listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No story selected.")
        return

    selected_index = selected_index[0]  

    if selected_index >= len(stories):  
        messagebox.showerror("Error", "Error retrieving the full story. Please regenerate the stories.")
        return

    selected_story = stories[selected_index]  

    try:
        if language == "english":
            
            filename = os.path.join(STORY_SAVE_PATH, f"story_english.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(selected_story)
            
            messagebox.showinfo("Success", f"English story saved:\n{filename}")

        elif language == "kannada":
            
            kannada_story = translator.translate(selected_story)  # Translate using deep_translator
            filename = os.path.join(STORY_SAVE_PATH, f"story_kannada.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(kannada_story)
            
            messagebox.showinfo("Success", f"Kannada story saved:\n{filename}")

        elif language == "both":
            
            kannada_story = translator.translate(selected_story)  

            english_filename = os.path.join(STORY_SAVE_PATH, f"story_english.txt")
            kannada_filename = os.path.join(STORY_SAVE_PATH, f"story_kannada.txt")

            with open(english_filename, "w", encoding="utf-8") as file:
                file.write(selected_story)

            with open(kannada_filename, "w", encoding="utf-8") as file:
                file.write(kannada_story)
            
            messagebox.showinfo("Success", f"Both English & Kannada stories saved:\n{english_filename}\n{kannada_filename}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save story: {e}")



# Create Main Window
window = tk.Tk()
window.title("Image to Story Generator")
window.geometry("500x600")

# Create a Main Frame to Hold Canvas and Scrollbar
main_frame = tk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a Canvas for Scrolling
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a Scrollbar Linked to the Canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

# Create Scrollable Frame Inside Canvas
content_frame = tk.Frame(canvas)
canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Function to Update Scroll Region
def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

content_frame.bind("<Configure>", update_scroll_region)

# Enable Mouse Wheel Scrolling
def on_canvas_scroll(event):
    canvas.yview_scroll(-1 * (event.delta // 120), "units")

canvas.bind_all("<MouseWheel>", on_canvas_scroll)

# 🔹 **Widgets Inside Scrollable Frame**
image_label = tk.Label(content_frame)
image_label.pack()

caption_label = tk.Label(content_frame, text="Caption: ", font=("Arial", 12))
caption_label.pack()

# Upload Section
upload_frame = tk.Frame(content_frame)
upload_frame.pack(pady=5)


# Upload Button with Arrow
upload_arrow_label = tk.Label(upload_frame, text="→", font=("Arial", 14))
upload_arrow_label.grid(row=0, column=1, padx=10)


upload_button = tk.Button(upload_frame, text="Upload Image", command=lambda: toggle_upload_buttons())
upload_button.grid(row=0, column=0, padx=10)

# Buttons that will appear on the right after the "Upload Image" button is clicked
upload_from_device_button = tk.Button(upload_frame, text="From Device", command=lambda: upload_image("device"))
upload_from_folder_button = tk.Button(upload_frame, text="From Folder", command=lambda: upload_image("folder"))

# Configure the grid columns so that they stay aligned
upload_frame.grid_columnconfigure(0, weight=1)  # The first column (Upload Button) takes available space
upload_frame.grid_columnconfigure(1, weight=0)  # The second column (Arrow) has no extra space
upload_frame.grid_columnconfigure(2, weight=0)  # The third column (From Device) has no extra space
upload_frame.grid_columnconfigure(3, weight=0)  # The fourth column (From Folder) has no extra space

def toggle_upload_buttons():
    """Toggle the visibility of upload options and the arrow direction."""
    # Check if the buttons are already in the grid
    if upload_from_device_button.winfo_ismapped():
        # If the buttons are already visible, hide them and change the arrow to right
        upload_from_device_button.grid_forget()
        upload_from_folder_button.grid_forget()
        upload_arrow_label.config(text="")  # Change arrow to right
    else:
        # If the buttons are not visible, show them and change the arrow to down
        upload_from_device_button.grid(row=0, column=2, padx=10, pady=2)  # Appears next to the upload button
        upload_from_folder_button.grid(row=0, column=3, padx=10, pady=2)  # Appears next to the upload button
        upload_arrow_label.config(text="→")  # Change arrow to down


tk.Button(content_frame, text="Capture Image", command=capture_image).pack()

# Story Generation Section
story_count_spinbox = tk.Spinbox(content_frame, from_=1, to=10)
story_count_spinbox.pack(pady=5)

tk.Button(content_frame, text="Generate Stories", command=generate_stories).pack()

story_listbox = tk.Listbox(content_frame, height=5)
story_listbox.pack()

story_text = tk.Text(content_frame, height=5, wrap=tk.WORD)
story_text.pack()

kannada_text = tk.Text(content_frame, height=5, wrap=tk.WORD)
kannada_text.pack()

            
def update_efficiency_label():
    if efficiency_label.winfo_exists():  # ✅ Check if the widget exists before updating
        efficiency_label.config(text="Updated Efficiency Metrics!")
# 🏆 Efficiency Metrics Display
efficiency_label = tk.Label(content_frame, text="Efficiency Metrics: ", wraplength=400, justify="center", font=("Arial", 12, "bold"))
efficiency_label.pack(pady=10)

tk.Button(content_frame, text="Read English Story", command=read_english_story).pack(pady=2)
tk.Button(content_frame, text="Read Kannada Story", command=lambda: read_kannada_story(kannada_text)).pack(pady=2)

# Save Buttons Frame
save_frame = tk.Frame(content_frame)
save_frame.pack(pady=5)

# Save Audio Section
save_audio_button = tk.Button(save_frame, text="Save Audio", command=lambda: toggle_save_audio_buttons())
save_audio_button.grid(row=0, column=0, padx=10)
# Upload Button with Arrow
save_audio_arrow_label = tk.Label(save_frame, text="→", font=("Arial", 14))
save_audio_arrow_label.grid(row=0, column=1, padx=10)

record_english = tk.Button(save_frame, text="Record English", command=lambda: record_audio("english"))
record_kannada = tk.Button(save_frame, text="Record Kannada", command=lambda: record_audio("kannada"))
record_both = tk.Button(save_frame, text="Record Both", command=lambda: record_audio("both"))


# Configure the grid columns so that they stay aligned
save_frame.grid_columnconfigure(0, weight=1)  # The first column (Upload Button) takes available space
save_frame.grid_columnconfigure(1, weight=0)  # The second column (Arrow) has no extra space
save_frame.grid_columnconfigure(2, weight=0)  # The third column (From Device) has no extra space
save_frame.grid_columnconfigure(3, weight=0)  # The fourth column (From Folder) has no extra space
save_frame.grid_columnconfigure(4, weight=0)

def toggle_save_audio_buttons():
    # Check if the buttons are already visible
    if record_english.winfo_ismapped():
        # If the buttons are already visible, hide them and change the arrow to right
        record_english.grid_forget()
        record_kannada.grid_forget()
        record_both.grid_forget()
        save_audio_arrow_label.config(text="")  # Change arrow to right
    else:
        # If the buttons are not visible, show them and change the arrow to down
        record_english.grid(row=0, column=2, padx=10, pady=2)  # Appears next to the save audio button
        record_kannada.grid(row=0, column=3, padx=10, pady=2)  # Appears next to the save audio button
        record_both.grid(row=0, column=4, padx=10, pady=2)  # Appears next to the save audio button
        save_audio_arrow_label.grid(row=0, column=1, padx=10)  # Show the arrow next to the save audio button
        save_audio_arrow_label.config(text="→")  # Change arrow to down




# Save Story Section
save_story_button = tk.Button(save_frame, text="Save Story", command=lambda: toggle_save_story_buttons())
save_story_button.grid(row=2, column=0, padx=10)

# Upload Button with Arrow
save_story_arrow_label = tk.Label(save_frame, text="→", font=("Arial", 14))
save_story_arrow_label.grid(row=2, column=1, padx=10)

save_story_english = tk.Button(save_frame, text="English", command=lambda: save_story("english"))
save_story_kannada = tk.Button(save_frame, text="Kannada", command=lambda: save_story("kannada"))
save_story_both = tk.Button(save_frame, text="Both", command=lambda: save_story("both"))

# Configure the grid columns so that they stay aligned
save_frame.grid_columnconfigure(0, weight=1)  # The first column (Upload Button) takes available space
save_frame.grid_columnconfigure(1, weight=0)  # The second column (Arrow) has no extra space
save_frame.grid_columnconfigure(2, weight=0)  # The third column (From Device) has no extra space
save_frame.grid_columnconfigure(3, weight=0)  # The fourth column (From Folder) has no extra space
save_frame.grid_columnconfigure(4, weight=0)

def toggle_save_story_buttons():
    # Check if the buttons are already visible
    if save_story_english.winfo_ismapped():
        # If the buttons are already visible, hide them and change the arrow to right
        save_story_english.grid_forget()
        save_story_kannada.grid_forget()
        save_story_both.grid_forget()
        save_story_arrow_label.config(text="")  # Change arrow to right
    else:
        # If the buttons are not visible, show them and change the arrow to down
        save_story_english.grid(row=2, column=2, padx=10, pady=2)  # Appears next to the save audio button
        save_story_kannada.grid(row=2, column=3, padx=10, pady=2)  # Appears next to the save audio button
        save_story_both.grid(row=2, column=4, padx=10, pady=2)  # Appears next to the save audio button
        save_story_arrow_label.grid(row=2, column=1, padx=10)  # Show the arrow next to the save audio button
        save_story_arrow_label.config(text="→")  # Change arrow to down


# Run Tkinter Main Loop
window.mainloop()
