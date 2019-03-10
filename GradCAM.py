from model.resnet import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import tqdm
import numpy as np
import cv2

def load_train_test(args):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)  

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2)  

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    cv2.imwrite("img.jpg", np.uint8(255 * img))

def grad_cam(dataloader, net, args, criterion, index=None):
    running_loss = 0.0
    pbar = tqdm.tqdm(enumerate(dataloader, 0))
    for i, data in pbar:
        # get the inputs
        inputs, labels = data
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        outputs = net(inputs)

        if index == None:
            index = np.argmax(outputs.detach().numpy(), axis=1)

        one_hot = np.zeros((outputs.shape), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()
        if args.cuda:
            one_hot = torch.sum(one_hot.cuda() * outputs)
        else:
            one_hot = torch.sum(one_hot * outputs)
        net.zero_grad()
        one_hot.backward()
        grads_val = net.grad_cam3[-1].detach().numpy()
        target = net.actv_cam3[-1].detach().numpy()
        
        IMG_IDX = 1

        target = target[IMG_IDX, :]
        weights = np.mean(grads_val, axis=(2, 3))[IMG_IDX, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        img = inputs.detach().numpy()[IMG_IDX, :, :, :] * 0.2 + 0.48
        img = np.transpose(img, (1, 2, 0))
        print(index[IMG_IDX])
        show_cam_on_image(img, cam)
        return cam

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--batch-size', type=int, default=192, metavar='N', 
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='WD',
                        help='weigth decay')
    parser.add_argument('--lr-decay-epoch', type=int, default=100, metavar='LDE',
                        help='number of epochs to decay lr')
    parser.add_argument('--save-path', type=str, default='',
                        help='save model to path')
    parser.add_argument('--load-path', type=str, default='',
                        help='load model from path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    return args

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    args = parse_args()
    trainloader, testloader = load_train_test(args)
    net = resnet50(num_classes=10)

    if args.load_path != '':
        if args.cuda:
            net.load_state_dict(torch.load(args.load_path))
        else:
            net.load_state_dict(torch.load(args.load_path, map_location='cpu'))
        if args.cuda:
            net = torch.nn.DataParallel(net).cuda()
        grad_cam(testloader, net, args, nn.CrossEntropyLoss())
        if args.cuda:
            print(net.module.grad_cam4[0].shape)
            print(len(net.module.gradient_cam))
        else:
            print(net.grad_cam4[0].shape)
            print(len(net.grad_cam4))
        return

    if args.cuda:
        #net.cuda()
        net = torch.nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        train(epoch, trainloader, net, criterion, optimizer, args, lr)
        test(testloader, net, args)
        if args.save_path != '':
            torch.save(net.module.state_dict(), args.save_path)

if __name__ == '__main__':
    main()


