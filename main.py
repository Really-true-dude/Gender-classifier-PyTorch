import torch, torch.nn
import torchvision
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from model import MaleFemaleClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "Classifier_Model_weights.pth"
model = MaleFemaleClassifier(input_shape = 3,
                             hidden_units = 10,
                             output_shape = 2).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

class_names = ["Female", "Male"]
img = ""
transform = transforms.Compose([
    transforms.Resize(size=(218, 178)),
])


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    
    basewidth = 150 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    
    img = img.resize((basewidth, hsize) ) # Image.ANTIALIAS
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    image = torchvision.io.read_image(str(image_data)).type(torch.float32)
    image = transform(image)
    image /= 255
    

    model.to(device)
    model.eval()

    with torch.inference_mode():
        image = image.unsqueeze(dim = 0)
        image_logits = model(image.to(device))

    image_pred_prob = torch.softmax(image_logits, dim = 1)
    image_pred_label = torch.argmax(image_pred_prob, dim = 1)
    
    result = tk.Label(frame,
                      text = f"Image of: {class_names[image_pred_label]} | Probability: {image_pred_prob.max():.3f}")
    result.pack()


if __name__ == "__main__":

    root = tk.Tk()
    root.title("Custom Gender Classifier")
    root.resizable(False, False)

    tit = tk.Label(root, text = "Custom Gender Classifier", padx = 25, pady = 6, font=("", 12)).pack()    

    canvas = tk.Canvas(root, height = 500, width = 500, bg ="grey")
    canvas.pack()

    frame = tk.Frame(root, bg = "white")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    choose_image = tk.Button(root, text="Choose Image",
                             padx = 35, pady = 10,
                             fg = "white", bg = "grey", command=load_img)
    choose_image.pack(side=tk.LEFT)

    classify_image = tk.Button(root, text="Classify Image",
                               padx = 35, pady = 10,
                               fg = "white", bg = "grey", command=classify)
    classify_image.pack(side=tk.RIGHT)

    root.mainloop()