from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from models import Generator  # Ensure models.py contains the Generator definition
from PIL import Image
import io
import torchvision.transforms as transforms
import base64
import os

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Generators
netG_A2B = Generator(input_nc=3, output_nc=3).to(device)
netG_B2A = Generator(input_nc=3, output_nc=3).to(device)
 
# Load model weights
netG_A2B.load_state_dict(torch.load('no_seg_best_netG_A2B.pth', map_location=device))
netG_B2A.load_state_dict(torch.load('no_seg_best_netG_B2A.pth', map_location=device))

# Set models to evaluation mode
netG_A2B.eval()
netG_B2A.eval()

# Define the transformation with normalization
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization
])

# Function to convert tensor to base64 string
def tensor_to_base64(tensor):
    tensor = tensor * 0.5 + 0.5  # De-normalize the image
    image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", quality=95)  # Adjust quality if needed
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Endpoint to serve HTML
@app.get("/", response_class=HTMLResponse)
async def main():
    file_path = "index.html"
    if not os.path.exists(file_path):
        return HTMLResponse(content="index.html not found", status_code=404)
    with open(file_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Endpoint to transform an image from A to B
@app.post("/transform/")
async def transform(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')
    img_tensor = transform_pipeline(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = netG_A2B(img_tensor).to(device)
    output_base64 = tensor_to_base64(output_tensor)
    input_base64 = tensor_to_base64(img_tensor)
    return JSONResponse(content={"input_image": input_base64, "transformed_image": output_base64})

# Endpoint to transform an image from B to A
@app.post("/transform2/")
async def transform2(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert('RGB')
    img_tensor = transform_pipeline(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = netG_B2A(img_tensor).to(device)
    output_base64 = tensor_to_base64(output_tensor)
    input_base64 = tensor_to_base64(img_tensor)
    return JSONResponse(content={"input_image": input_base64, "transformed_image": output_base64})

# Run the application with: uvicorn app:app --reload --port 8001
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)