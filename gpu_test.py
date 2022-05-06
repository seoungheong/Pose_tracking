import torch

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print(torch.cuda.is_available())
    print(torch.__version__)

# 3090 RTX 설정
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113