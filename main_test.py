from model import CharCNN
import torch

model = CharCNN(70, 0.5)
model.load_state_dict(torch.load('save_model/best.pt'))

sent = "U.S. Brokers Cease-fire in Western Afghanistan KABUL (Reuters) - The United States has brokered a  cease-fire between a renegade Afghan militia leader and the  embattled governor of the western province of Herat,  Washington's envoy to Kabul said Tuesday."
sent_tensor = torch.zeros(1014).long()
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
for i, char in enumerate(sent):
    if i == 1014:
        break
    alphabet_index = alphabet.find(char)
    if alphabet_index != -1:
        sent_tensor[i] = alphabet_index

sent_tensor = sent_tensor.view(-1, sent_tensor.size(0))
out_feature = model(sent_tensor)
out_feature = out_feature.squeeze(0)
print('out_feature:', out_feature)