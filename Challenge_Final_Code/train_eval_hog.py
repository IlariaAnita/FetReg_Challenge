import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


'''SEGMENTATION MULTI-CLASS'''

def train_seg(model, iterator, optimizer, criterion, device):
    scaler = GradScaler()

    epoch_loss = 0

    model.train()

    for l, (input_train, target_train, hog_image, hog_vector) in enumerate(tqdm(iterator)):
        input_train = input_train.to(device, non_blocking=True)
        target_train = target_train.to(device, torch.long, non_blocking=True).squeeze_(1)
        hog_vector = hog_vector.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            output = model(input_train, hog_vector)
            loss = criterion(output, target_train)
        #loss.backward()

        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate_seg(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    for l, (input_val, target_val, hog_image, hog_vector) in enumerate(tqdm(iterator)):
        input_val = input_val.to(device, non_blocking=True)
        target_val = target_val.to(device, torch.long, non_blocking=True).squeeze_(1)
        hog_vector = hog_vector.to(device, non_blocking=True)

        with torch.no_grad():
            scores = model(input_val, hog_vector)
            loss = criterion(scores, target_val)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)



