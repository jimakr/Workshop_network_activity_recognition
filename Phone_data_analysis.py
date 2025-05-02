import numpy as np
import torch
import pytorch_lightning as pl
from Networks.MLP_activity import LitMLP

# Step 1: Load the .npz data
phone_data = np.load('./Dataset/phone_traffic.npz')['phone_data']
# Step 2: Convert numpy array to torch tensor
tensor_data = torch.from_numpy(phone_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_data = tensor_data.to(device)

# Step 3: Load the trained Lightning model from checkpoint
# Make sure you have the exact class definition of your model
 # replace with your actual model class and file

model = LitMLP.load_from_checkpoint('./Networks/checkpoints/MLP_activity/last.ckpt')

# Step 4: Set model to eval mode and move to device
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tensor_data = tensor_data.to(device).float()

# Step 5: Predict row by row
predictions = []
with torch.no_grad():
    for row in tensor_data:
        row = row.unsqueeze(0)  # batch size of 1
        pred = model(row)
        predictions.append(pred.cpu().numpy())

predictions = np.array(predictions).reshape(-1,23).argmax(axis=1)
print(predictions)


class_map = ['Background', 'ClashRoyale', 'Crunchyroll', 'Discord', 'GotoMeeting', 'JitsiMeet', 'KakaoTalk', 'Line', 'Meet', 'Messenger', 'Omlet', 'Playstore', 'Signal', 'Skype', 'Slack', 'Teams', 'Telegram', 'Trueconf', 'Twitch', 'Webex', 'WhatsApp', 'Workplace', 'Zoom']
class_map = {i:class_name for i,class_name in enumerate(class_map)}

predictions2 = [class_map[i] for i in predictions]
print(predictions2)