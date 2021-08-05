import torch
import torch.nn as nn

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

    # 테스트 데이터 입력 후 출력 확인
    test_input = torch.randn(1, 3, 224, 224)
    out = model(test_input)
    print(f"{out.shape} \n {out}")

    print("Model is Ready")