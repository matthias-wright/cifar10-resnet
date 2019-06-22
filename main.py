from resnet import ResNet


net = ResNet()
net.load_parameters(path='saves/resnet__epoch_75.pth')
net.train(save_dir='saves', num_epochs=75, batch_size=256, learning_rate=0.001, verbose=True)
accuracy = net.test()
print('Test accuracy: {}'.format(accuracy))
