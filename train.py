import torch
import argparse

from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a network on a dataset of images and saves the model to a checkpoint")
    parser.add_argument('data_dir', type=str, help='set the data directory')
    parser.add_argument('--arch', default = 'vgg16', type=str, help='choose the model architecture')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--hidden_units', default=25088, type=int, help='the sizes of the hidden layers')
    parser.add_argument('--epochs', default=3, type=int, help='number of training epochs')
    parser.add_argument('--gpu', help='set the gpu mode')
    parser.add_argument('--save_dir', default = '', type=str, help='set the checkpoint path')
    args = parser.parse_args()
    return args

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer,device):
    print("Deep learning started...")
    epochs = epochs
    print_every = print_every
    steps = 0
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss  = 0
                for ii, (inputs, labels) in enumerate(validloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    validation_loss  += criterion(output, labels)
                    probabilities = torch.exp(output).data
                    equality = (labels.data == probabilities.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      " Training Loss: {:.4f}".format(running_loss / print_every),
                      " Validation Loss: {:.3f}.. ".format(validation_loss  / len(testloader)),
                      " Validation Accuracy: {:.3f}%".format(accuracy / len(testloader) * 100))



                running_loss = 0
                mode.train()
    print("Model trained")
    
def build_classifier(model_name,model,hidden_units,output_size):
    
    if 'vgg' in model_name:
        filters = model.classifier[0].in_features
    elif 'densenet' in model_name:
        filters = model.classifier.in_features
    elif 'resnet' in model_name:
        filters = model.fc.in_features
    
    classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(filters, hidden_units)),
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(p=0.1)),
                         ('fc2', nn.Linear(hidden_units, output_size)),
                         ('output', nn.LogSoftmax(dim=1))
                         ]))
    return classifier

def main():
    args = parse_args()
    data_dir = args.data_dir
    gpu = args.gpu
    model_name = args.arch
    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    checkpoint_dir = args.save_dir
    
    print('Data directory:      {}'.format(data_dir))
    print('Model:         {}'.format(model_name))
    print('Hidden layers: {}'.format(hidden_units))
    print('Learning rate: {}'.format(lr))
    print('Epochs:        {}'.format(epochs))
    print('Checkpoint directory:        {}'.format(checkpoint_dir))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    print_every = 40
     

    train_transforms = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])
    validation_transforms = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                ])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(test_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    user_model = getattr(models,model_name)
    model = user_model(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = build_classifier(model_name,model,hidden_units,102)
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device)

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size':hidden_units,
              'model': model,
              'model_name': model_name,
              'lr': lr,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier' : model.classifier,
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, './' + checkpoint_dir + '/checkpoint.pth')
    
if __name__ == '__main__':
    main()
