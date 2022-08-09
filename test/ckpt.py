import torch

a = torch.load('running_log/yolo_ch/last_epoch.pth')
print(a.keys())
for key in a['model'].keys():
    print(key)
torch.save(a['model'], 'yolov3_load_templete.pt')