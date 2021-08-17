import os
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler

from dataset import image_folder_dataset
from vgg_model import VGG

# CUDA 를 활용한 GPU 가속 여부에 따라, 장치를 할당 할 수 있도록 변수로 선언
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(parameters):
    # Prepare Image DataSets
    train_image_dir = parameters['datasets'][0]
    eval_img_dir = parameters['datasets'][1]
    image_dataset = image_folder_dataset(train_image_dir, eval_img_dir,
                                         batch_size=parameters['batch_size'])

    # Train & Eval Dataloader
    train_dataloader = image_dataset['train_dataloader']
    eval_dataloader = image_dataset['eval_dataloader']
    class_index = image_dataset['class_index']
    train_steps = len(train_dataloader)
    eval_steps = len(eval_dataloader)

    # Prepare Custom Model
    model = VGG(parameters['vgg11_cfg'], num_classes=len(class_index), init_weights=True)  # model 빌드
    model.to(device)   # 모델의 장치를 device 에 할당
    model.zero_grad()  # 모델 gradient 초기화
    model.train()      # Train 모드로 모델 설정

    # Loss Function
    criterion = nn.CrossEntropyLoss()  # loss function

    # Optimizer & LR_Scheduler setting
    optimizer = Adam(model.parameters(),  # Adam 옵티마이저 세팅
                     lr=parameters['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer,  # 선형 스케줄러 세팅 - 학습률 조정용 스케줄러
                                    parameters['scheduler_step'],
                                    parameters['scheduler_gamma'])

    # Train Start
    train_iterator = trange(int(parameters['epoch']), desc="Epoch")  # 학습 상태 출력을 위한 tqdm.trange 초기 세팅
    global_step = 0

    # Epoch 루프
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc='epoch: X/X, global: XXX/XXX, tr_loss: XXX'  # Description 양식 지정
        )
        epoch = epoch + 1

        # Step(batch) 루프
        for step, batch in enumerate(epoch_iterator):
            # 모델이 할당된 device 와 동일한 device 에 연산용 텐서 역시 할당 되어 있어야 함
            image_tensor, tags = map(lambda elm: elm.to(device), batch)  # device 에 연산용 텐서 할당
            out = model(image_tensor)      # Calculate
            loss = criterion(out, tags)    # loss 연산

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # Update learning rate schedule
            global_step += 1
            # One train step Done

            # Step Description
            epoch_iterator.set_description(
                'epoch: {}/{}, global: {}/{}, tr_loss: {:.3f}'.format(
                    epoch, parameters['epoch'],
                    global_step, train_steps * parameters['epoch'],
                    loss.item()
                )
            )

        # -- Evaluate & Save model result -- #
        # 한 Epoch 종료 시 평가, 평가 결과 정보를 포함한 이름으로 학습된 모델을 지정된 경로에 저장
        eval_result = evaluate(model, criterion, eval_dataloader)
        # Set Save Path
        os.makedirs(parameters['train_output'], exist_ok=True)
        save_path = parameters['train_output'] + \
                    f"/epoch-{epoch}-acc-{eval_result['mean_acc']}-loss-{eval_result['mean_loss']}"
        # Save
        torch.save(model.state_dict(), save_path + "-model.pth")

def evaluate(model, criterion, eval_dataloader):
    # Evaluation
    model.eval()  # 모델의 AutoGradient 연산을 비활성화하고 평가 연산 모드로 설정 (메모리 사용 및 연산 효율화를 위해)
    sum_eval_acc, sum_eval_loss = 0, 0
    eval_result = {"mean_loss": 0, "mean_acc": 0}

    eval_iterator = tqdm(  # Description 양식 지정
        eval_dataloader, desc='Evaluating - mean_loss: XXX, mean_acc: XXX'
    )

    # Evaluate
    for e_step, e_batch in enumerate(eval_iterator):
        image_tensor, tags = map(lambda elm: elm.to(device), e_batch)  # device 에 연산용 텐서 할당
        out = model(image_tensor)  # Calculate
        loss = criterion(out, tags)

        # Calculate acc & loss
        sum_eval_acc += (out.max(dim=1)[1] == tags).float().mean().item()  # 정답과 추론 값이 일치하는 경우 정답으로 count
        sum_eval_loss += loss.item()

        # 평가 결과 업데이트
        eval_result.update({"mean_loss": sum_eval_acc / (e_step + 1),
                            "mean_acc": sum_eval_loss / (e_step + 1)})

        # Step Description
        eval_iterator.set_description(
            'Evaluating - mean_loss: {:.3f}, mean_acc: {:.3f}'.format(
                eval_result['mean_loss'], eval_result['mean_acc'])
        )
    
    model.train()  # 평가 과정이 모두 종료 된 뒤, 다시 모델을 train 모드로 변경

    return eval_result

if __name__ == "__main__":
    # Train Parameter
    parameters = {
        "datasets": ["datasets/train", "datasets/evaluation"],                            # 학습용 데이터 경로
        "vgg11_cfg": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],    # VGG 모델의 구조 정보
        "class_info_file": "datasets/class_index.json",                                   # 학습용 데이터의 class_info_file
        "epoch": 10,                # 전체 학습 Epoch
        "batch_size": 16,           # Batch Size
        "learning_rate": 0.003,     # 학습률
        "scheduler_step": 100,      # 어느정도 step 주기로 학습률을 감소할지 지정
        "scheduler_gamma": 0.9,     # 학습률의 감소 비율
        "train_output": "output"    # 학습된 모델의 저장 경로
    }
    train(parameters)  # train

    print("Train Complete")