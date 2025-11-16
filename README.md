# Reconhecimento de Dígitos com PyTorch

Este projeto treina uma rede neural simples para reconhecer dígitos manuscritos usando o dataset MNIST. A interface interativa foi construída com Gradio e está disponível online via Hugging Face Spaces.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-🧠-red)
![Gradio](https://img.shields.io/badge/Gradio-UI-green)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

---

## Como rodar localmente

```bash
git clone https://github.com/tineslee/rede-neural-pytorch.git
cd rede-neural-pytorch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```

---
## Estrutura do projeto
rede-neural-pytorch/

├── app.py               # Interface Gradio

├── train.py             # Treinamento da rede neural

├── test.py              # Testes e validações

├── requirements.txt     # Dependências

├── models/              # Modelo treinado (.pth)

└── .gradio/flagged/     # Dados salvos pela interface

 ## Resultados

Acurácia no conjunto de teste: 97.16%

## Demo online

👉 [Teste a demo online](https://huggingface.co/spaces/tinesslee/rede-neural-pytorch)

## Tecnologias usadas

PyTorch

Gradio

MNIST Dataset

Python

## Autor

Feito com 💙 por Thais Inês


