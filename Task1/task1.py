from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from arch17 import Model

app = Flask(__name__)
model = Model()
model.load_state_dict(torch.load('/Users/ali/Documents/Apidon/API/api-tutorial/Task1/model_23.pth'))
model.eval()

@app.route('/classify', methods=['POST'])
def classify_image():
    image_url = request.json['image_url']
    # Load and preprocess the image
    image = Image.open(image_url)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Use the model to make predictions
    with torch.no_grad():
        output = model(input_batch)
    predictions = output.tolist()[0]

    return jsonify(predictions_array=predictions)

if __name__ == '__main__':
    app.run(debug=True)
