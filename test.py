import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw

def relu(x):
    return np.maximum(0,x)

def softmax1(x):
    x=x-np.max(x)
    ex=np.exp(x)
    return ex/np.sum(ex)

def testing(inputList,w1,w2,w3,w4,b1,b2,b3,b4):
    a1=relu(np.dot(inputList,w1)+b1)
    a2=relu(np.dot(a1,w2)+b2)
    a3=relu(np.dot(a2,w3)+b3)
    output=softmax1(np.dot(a3,w4)+b4)
    return np.argmax(output)

class DigitRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("400x500")
        self.root.configure(bg='#f0f0f0')
        
        # Load weights
        weights = np.load('weights.npz')
        self.w1, self.w2, self.w3, self.w4 = weights['w1'], weights['w2'], weights['w3'], weights['w4']
        self.b1, self.b2, self.b3, self.b4 = weights['b1'], weights['b2'], weights['b3'], weights['b4']
        
        # Title
        title = tk.Label(self.root, text="Draw a Digit (0-9)", font=("Arial", 18, "bold"), bg='#f0f0f0')
        title.pack(pady=10)
        
        # Canvas for drawing
        self.canvas = Canvas(self.root, width=280, height=280, bg='black', bd=2, relief='solid', cursor='pencil')
        self.canvas.pack(pady=10)
        
        # Result label
        self.result_label = tk.Label(self.root, text="Draw a digit", font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333')
        self.result_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Buttons
        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict, 
                               font=("Arial", 12, "bold"), bg='#4CAF50', fg='white', 
                               padx=20, pady=5, relief='raised')
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas,
                             font=("Arial", 12, "bold"), bg='#f44336', fg='white',
                             padx=20, pady=5, relief='raised')
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint_pressed)
        self.canvas.bind("<Button-1>", self.paint_pressed)
        self.canvas.bind("<ButtonRelease-1>", self.paint_released)
        
        self.is_pressed = False
        
        # Image for processing
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
    
    def paint_pressed(self, event):
        self.is_pressed = True
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def paint_released(self, event):
        self.is_pressed = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit", fg='#333')
    
    def predict(self):
        # Save the drawn image
        import time
        timestamp = int(time.time())
        
        # Crop to bounding box of drawn content with margin
        img_array = np.array(self.image)
        coords = np.column_stack(np.where(img_array > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add margin
            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(280, x_max + margin)
            y_max = min(280, y_max + margin)
            cropped = self.image.crop((x_min, y_min, x_max+1, y_max+1))
        else:
            cropped = self.image
        
        # Resize cropped image to 28x28, convert to grayscale and save
        img_resized = cropped.resize((28, 28)).convert('L')
        img_resized.save(f"drawn_digit_{timestamp}.png")
        
        # Process exactly like training data
        img_array = np.array(img_resized).flatten()
        img_array = [x / 255 for x in img_array]  # Match training format
        
        # Predict
        prediction = testing(img_array, self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4)
        self.result_label.config(text=f"Prediction: {prediction}", fg='#2196F3')
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitRecognizer()
    app.run()