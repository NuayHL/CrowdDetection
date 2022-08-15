import torch

a = torch.load('YOLOv3_siou_e280.pth')
print(a.keys())
for key in a['model'].keys():
    print(key)
torch.save({'model':a['model']}, 'YOLOv3_640.pth')