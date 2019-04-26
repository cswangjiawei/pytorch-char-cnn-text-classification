import torch


def train_model(dataloader, model, criterion, optimizer):
    model.train()
    total_loss = 0.

    for batch in dataloader:
        model.zero_grad()
        batch_sent = batch['sent']
        batch_label = batch['label'].view(-1)
        batch_label.sub_(1)
        out_feature = model(batch_sent)
        loss = criterion(out_feature, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(dataloader, model):
    model.eval()
    wrong_num = 0
    total_num = 0

    for batch in dataloader:
        batch_sent = batch['sent']
        batch_label = batch['label'].view(-1)
        batch_label.sub_(1)
        out_feature = model(batch_sent)
        _, preds = torch.max(out_feature, 1)
        wrong_num += torch.sum((preds != batch_label)).item()
        total_num += len(batch_label)

    test_error = (wrong_num / total_num) * 100
    return test_error