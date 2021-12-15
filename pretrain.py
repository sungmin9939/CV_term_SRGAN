import torch
import torch.nn as nn
from torch._C import dtype
from models.basics import Generator, Discriminator
from dataloader import TrainData
from loss import VGGLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.transforms as transforms
import csv,os


def train():
    #Initialize each model
    gen = Generator()
    if os.path.exists('./result/Generator.pth'):
        gen.load_state_dict(torch.load('./result/Generator.pth'))

    # move models to GPU
    device = torch.device('cuda:0')
    gen = gen.to(device)

    #set models to train mode
    gen.train()

    #initialize dataloader
    trainset = TrainData('./dataset')
    trainloader = DataLoader(trainset,16,True)
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

    #initialize loss function
    gen_criterion = nn.MSELoss()    

    #set optimizer
    optimG = optim.Adam(gen.parameters(),lr=0.0001, betas=(0.9, 0.999))

    epoch = 10

    for i in range(epoch):
        #training result summary

        trainloader = tqdm(trainloader)
        train_result = {'content_loss': 0}


        for idx, (target, low) in enumerate(trainloader):
            target = Variable(normalize(target)).to(device)
            low = Variable(normalize(low)).to(device)

            gen.zero_grad()
            fake_target = gen(low)

            content_loss = gen_criterion(target, fake_target)
            content_loss.backward()
            optimG.step()

            train_result['content_loss'] += content_loss


            #update Discriminator using true HR images(target) and SR images generated from low

            

            
        
            

    
        print(train_result)
        
        file = open('./result/pretrained_gen.csv','a')
        records = csv.writer(file)
        records.writerow([train_result['content_loss'].item()])
        file.close()
        torch.save(gen.state_dict(),'./result/Generator.pth')









if __name__ == "__main__":
    train()