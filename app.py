import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

# Mesma arquitetura usada no treino
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# Carregar modelo treinado
model = MLP()
model.load_state_dict(torch.load("models/mlp_mnist.pth", map_location=torch.device("cpu")))
model.eval()

# Transformações para imagem
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Função de previsão
def predict(image):
    try:
        # Aqui a imagem já vem como PIL porque usamos type="pil"
        image = image.convert("L")  # força grayscale
        image = transform(image).unsqueeze(0)  # [1, 1, 28, 28]

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return f"Você desenhou/enviou um **{prediction}**!"
    
    except Exception as e:
        return f"Erro ao processar imagem: {str(e)}"

# Interface Gradio
gr.Interface(
    fn=predict,
    inputs=gr.Image(image_mode="L", type="pil"),
    outputs="text",
    title="Reconhecimento de Dígitos com PyTorch",
    description="Envie ou desenhe um número de 0 a 9 e veja a previsão feita pela rede neural."
).launch(inbrowser=True)




