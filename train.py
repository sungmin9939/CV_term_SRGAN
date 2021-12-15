import torch
import torch.nn as nn
from torch._C import dtype
from torch.random import get_rng_state
from torch.serialization import save
from models.basics import Generator, Discriminator
from dataloader import TrainData, testData
from loss import VGGLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import csv
import pytorch_ssim
from math import log10
import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize




def train():
    trainset = TrainData('./dataset')
    trainloader = DataLoader(trainset,16,True)

    testset = testData()
    testloader = DataLoader(testset, 1, False)
    
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

    generator = Generator()
    if os.path.exists('./result/Generator.pth'):
        generator.load_state_dict(torch.load('./result/Generator.pth'))

    discriminator = Discriminator()
    if os.path.exists('./result/Discriminator.pth'):
        discriminator.load_state_dict(torch.load('./result/Discriminator.pth'))

    vgg_criterion = VGGLoss()
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(16,1))

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        vgg_criterion.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()
    
    optimG = optim.Adam(generator.parameters(),lr=0.0001, betas=(0.9, 0.999))
    optimD = optim.Adam(discriminator.parameters(),lr=0.0001, betas=(0.9, 0.999))

    epoch = 1000
    save_img = False

    for i in range(epoch):
        if i % 10 == 0:
            save_img = True
            os.makedirs('./result/test_result/epoch{}'.format(i), exist_ok=True)
        else:
            save_img = False

        generator.train()
        discriminator.train()

        train_result = {'d_loss':0, 'g_loss':0, 'psnr': 0, 'ssim': 0}
        trainloader = tqdm(trainloader)

        for idx, (high_img, low_img) in enumerate(trainloader):
            high_img = normalize(high_img)
            low_img = normalize(low_img)

            
            high_img_real = Variable(high_img.cuda())
            high_img_fake = generator(Variable(low_img).cuda())
            target_real = Variable(torch.rand(16,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(16,1)*0.3).cuda() 

            ## train discriminator
            discriminator.zero_grad()

            d_loss = adversarial_criterion(discriminator(high_img_real), target_real) + \
                     adversarial_criterion(discriminator(Variable(high_img_fake.data)), target_fake)
            d_loss.backward()
            optimD.step()

            ## train generator
            generator.zero_grad()

            #g_content_loss = vgg_criterion(high_img_fake, high_img_real) + content_criterion(high_img_real, high_img_fake)
            g_content_loss = vgg_criterion(high_img_fake, high_img_real) + 0.01 * content_criterion(high_img_real, high_img_fake)
            g_adversarial_loss = adversarial_criterion(discriminator(high_img_fake), ones_const)

            g_loss = 0.001*g_adversarial_loss + g_content_loss
            g_loss.backward()
            optimG.step()

            train_result['d_loss'] += d_loss
            train_result['g_loss'] += g_loss


        generator.eval()

        with torch.no_grad():
            testloader = tqdm(testloader)
            testing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            test_images = []
            for idx, (hr, lr) in enumerate(testloader):
                batch_size = hr.size(0)
                testing_results['batch_sizes'] += batch_size
                hr = hr.cuda()
                lr = lr.cuda()
                sr = generator(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                testing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                testing_results['ssims'] += batch_ssim * batch_size
                testing_results['psnr'] = 10 * log10((hr.max()**2) / (testing_results['mse'] / testing_results['batch_sizes']))
                testing_results['ssim'] = testing_results['ssims'] / testing_results['batch_sizes']

                if save_img:
                    test_images.append((sr.data.cpu().squeeze(0)))
            if save_img:
                idx = 1
                for img in test_images:
                    utils.save_image(img, './result/test_result/epoch{}/epoch{}_index{}.png'.format(i, i, idx))
                    idx += 1
        
        save_img = False


        print(train_result)
        print(testing_results)

        file = open('./result/srgan_loss.csv','a')
        records = csv.writer(file)
        records.writerow([train_result['d_loss'].item(), train_result['g_loss'].item(), testing_results['psnr'],testing_results['ssim']])
        file.close()
        torch.save(generator.state_dict(),'./result/Generator.pth')
        torch.save(discriminator.state_dict(),'./result/Discriminator.pth')

        









if __name__ == "__main__":
    train()