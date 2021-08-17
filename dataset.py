# dataset.py

# 필요한 패키지 import
import json
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from vgg_model import VGG

def image_folder_dataset(train_img_dir, eval_img_dir, test_img_dir, batch_size=16):
    """
    Make Train & Evaluation Image-Dataset from Image-Folder
    :param train_img_dir: Train Image data directory (str)
    :param eval_img_dir: Evaluation Image data directory (str)
    :return: dict{
                      train_dataloader (torch.utils.data.DataLoader),
                      eval_dataloader (torch.utils.data.DataLoader),
                      class_index (dict)
                  }
    """

    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize(256),                             # 이미지의 크기를 256 으로 Resize
        transforms.CenterCrop(224),                         # 256 사이즈의 이미지 중앙에 정사각형 영역 (224 * 224 사이즈) 을 Crop
        transforms.ToTensor(),                              # RGB 3 채널에 대해 픽셀별로 0~1 사이의 값을 갖는 Float 형 Tensor 로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 입력된 mean, std 로 정규화 → 연산 효율을 올리기 위함 입니다. 정규화를 거치지 않아도 학습 가능
                             std=[0.229, 0.224, 0.225])     # 작성된 mean, std 는 ImageNet 으로부터 추출된 일반적인 이미지의 평균/표준편차 값
    ])
    # 하나의 이미지는 주어진 mean, std 값으로 정규화 된 (3, 224, 224) 사이즈의 Float 텐서로 변환
    # Tensor 차원 해석 : (RGB-Channel, Image-Height, Image-Width)

    # 이미지 폴더로부터 데이터 생성
    train_dataset = datasets.ImageFolder(root=train_img_dir,
                                         transform=preprocess)
    eval_dataset = datasets.ImageFolder(root=eval_img_dir,
                                        transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_img_dir,
                                        transform=preprocess)

    # train_dataset.classes 로 생성된 이미지 dataset 의 class 를 확인 가능
    # class 와 index 매칭을 위해 dict 형식으로 작성 및 json 파일로 저장
    class_index = {i: string for i, string in enumerate(train_dataset.classes)}
    with open("datasets/class_index.json", "w") as i:
        json.dump(class_index, i)
    with open("datasets/class_index.json", "r") as i:    # 저장 확인
        class_index = json.load(i)

    if not train_dataset.classes == eval_dataset.classes:    # 두 dataset 의 class 가 다른지 확인
        print("train & validation classes are not same")
        print(f"train.classes = {train_dataset.classes}")
        print(f"evaluation.classes = {eval_dataset.classes}")
        return None

    # DataLoader 생성
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,    # 임의의 batch_size
                                  shuffle=True)             # 학습 데이터를 랜덤하게 호출
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=batch_size,    # 임의의 batch_size
                                 shuffle=False)    # 평가시엔 랜덤하게 데이터를 호출할 필요가 없으므로 False
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,    # 임의의 batch_size
                                 shuffle=False)

    image_dataset = {
        "train_dataloader": train_dataloader,
        "eval_dataloader": eval_dataloader,
        "test_dataloader": test_dataloader,
        "class_index": class_index
    }

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()

    # 이미지 보여주기
    imshow(torchvision.utils.make_grid(images))
    # 정답(label) 출력
    print(' '.join('%5s' % labels[j] for j in range(4)))
    plt.show()
    model = VGG(nn.Module)
    model.load_state_dict(torch.load('./output/epoch-10-acc-0.9023336829508052-loss-0.5882352941176471-model.pth'))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    return image_dataset

if __name__ == "__main__":
    train_img_dir = "datasets/train"
    eval_img_dir = "datasets/evaluation"
    test_img_dir = "datasets/test"
    image_dataset = image_folder_dataset(train_img_dir, eval_img_dir, test_img_dir)

    # 생성된 dataset 확인
    for batch in image_dataset['train_dataloader']:
        input, target = batch
        print(f"input batch shape = {input.shape}")
        print(f"target batch = {target}")
        break    # 1 iter 만 확인하도록 break
    # >>> input batch shape = torch.Size([16, 3, 224, 224])
    # >>> target batch = tensor([0, 1, 1, 0, 1, 1, 2, 0, 0, 2, 2, 0, 1, 0, 1, 2])
    # dataloader 에서 배치 사이즈를 지정한 뒤, iterator 로 호출하면 input: torch.Size([batch, 3, 224, 224]), target: torch.Size([batch]) 로 batch tensor 출력

    print(f"class_index : {image_dataset['class_index']}")
    # >>> class_index : {'0': 'bird', '1': 'cat', '2': 'dog'}

    class VGG(nn.Module):
        def __init__(self, config, num_classes=10, init_weights=True):
            super(VGG, self).__init__()
            self.features = self._make_layers(config)    # VGG 모델의 구조 정보(cfg) 로부터, Feature 추출기 작성
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(             # 이미지에서 추출된 특징(Feature)으로부터 class 를 추측하는 분류기 작성
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(1024, num_classes),
            )
            if init_weights:                             # 가중치 초기화 로직
                self._initialize_weights()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        def _make_layers(self, cfg):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    if __name__ == "__main__":
        vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']    # VGG 모델의 구조 정보
        model = VGG(vgg11_cfg, num_classes=10, init_weights=True)    # model 빌드

    #테스트 데이터 입력 후 출력 확인
        test_input = torch.randn(1, 3, 224, 224)
        out = model(test_input)
        print(f"{out.shape} \n {out}")

        print("Model is Ready")

