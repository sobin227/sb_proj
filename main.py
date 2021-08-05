# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_dataset = torchvision.datasets.MNIST(root=f'{BASE_DIR}/datasets',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
eval_dataset = torchvision.datasets.MNIST(root=f'{BASE_DIR}/datasets',
                                          train=False,
                                          transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,  # 임의 지정 batch
                                           shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                          batch_size=64,   # 임의 지정 batch
                                          shuffle=False)


model = nn.Sequential(
    nn.Linear(784, 1024),  # 28 * 28 = 784
    nn.Linear(1024, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),  # 역전파 연산을 할 모델의 파라미터
    lr=0.003  # 임의의 learning_rate
)

# Model 학습
epochs = 10  # 전체 데이터를 모두 학습하는 epoch 를 몇번 반복할 것인지. 임의 값
total_step = len(train_loader)
model.train()  # 모델의 AutoGradient 연산을 활성화하는 학습 모드로 설정

# epoch 루프
for epoch in range(epochs):

    # step 루프
    for i, (inputs, targets) in enumerate(train_loader):
        # MNIST 텐서는 (batch, 28, 28) 의 형태이므로,
        # 테스트 모델에 적합하도록 (batch, 768) 의 형태로 Reshape 합니다
        inputs = inputs.reshape(-1, 28 * 28)  # 28 * 28 = 784

        # 순전파 - 모델의 추론 및 결과의 loss 연산
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()  # optimizer 초기화 (과거 학습 step 의 gradient 영향을 받지 않기 위해 필요)
        loss.backward()  # loss 의 역전파
        optimizer.step()  # 모델의 학습

        # 학습 상태 정보 출력
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

    # 한 epoch 가 모두 돈 뒤, Model 평가
    model.eval()  # 모델의 AutoGradient 연산을 비활성화하고 평가 연산 모드로 설정 (메모리 사용 및 연산 효율화를 위해)
    correct_cnt = 0
    total_cnt = 0
    for (inputs, targets) in eval_loader:
        # 학습 step 과 동일하게 추론 및 결과의 loss 연산을 진행
        inputs = inputs.reshape(-1, 28 * 28)  # 28 * 28 = 784
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)  # 가장 큰 값을 갖는 class index 가 모델이 추론한 정답
        total_cnt += targets.size(0)
        correct_cnt += (predicted == targets).sum()  # 정답과 추론 값이 일치하는 경우 정답으로 count

    print('Model Accuracy: {:.2f} %'.format(100 * correct_cnt / total_cnt))
    model.train()  # 평가가 모두 완료되었으므로 다시 학습 모드로 전환

# Model Checkpoint 저장
torch.save(model.state_dict(), 'mnist_dnn_model.pth')







#def print_hi(name):




# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
