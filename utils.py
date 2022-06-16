import torch
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,device):

    torch.cuda.empty_cache()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        training_acc = 0
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                model.train()
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)
                # forward
                output = model(image)
                loss = criterion(output, label)
                predict_t = torch.max(output, dim=1)[1]

                # backward
                loss.backward()
                optimizer.step()  # update weight

                running_loss += loss.item()
                training_acc += torch.eq(predict_t, label).sum().item()
                pbar.update(1)

        model.eval()
        val_losses = 0
        validation_acc = 0
        # validation loop
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    predict_v = torch.max(output, dim=1)[1]

                    val_losses += loss.item()
                    validation_acc += torch.eq(predict_v, label).sum().item()
                    pb.update(1)

            # calculatio mean for each batch
            train_loss.append(running_loss / len_train)
            val_loss.append(val_losses / len_val)

            train_acc.append(training_acc / len_train)
            val_acc.append(validation_acc / len_val)

            torch.save(model, "./weight/last.pth")
            if best_acc <(validation_acc / len_val):
                torch.save(model, "./weight/best.pth")


            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Acc: {:.3f}..".format(training_acc / len_train),
                  "Val Acc: {:.3f}..".format(validation_acc / len_val),
                  "Train Loss: {:.3f}..".format(running_loss / len_train),
                  "Val Loss: {:.3f}..".format(val_losses / len_val),
                  "Time: {:.2f}s".format((time.time() - since)))


    history = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history


def plot_loss(x, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.plot(x, history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight/loss.png')
    plt.show()


def plot_acc(x, history):
    plt.plot(x, history['train_acc'], label='train_acc', marker='x')
    plt.plot(x, history['val_acc'], label='val_acc', marker='x')
    plt.title('Acc per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('./weight/acc.png')
    plt.show()