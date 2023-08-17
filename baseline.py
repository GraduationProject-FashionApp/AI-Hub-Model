from __future__ import print_function, division

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet18_Weights
from torch.optim import lr_scheduler

# Train 파일 다운받기
# https://kr.object.ncloudstorage.com/aihub-competition/dataset/K-Fashion_Train.zip

def main():

    # 압축 해제할 디렉토리 경로 지정
    extract_path = 'D:\\AiHub\\K-Fashion_Train\\'

    # 이미지 데이터를 로드할 경로 설정
    data_dir = extract_path

    # 이미지 데이터 전처리를 위한 변환 정의
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),             # 이미지 무작위 크롭 및 크기 조정
        transforms.RandomHorizontalFlip(),             # 확률 0.5로 이미지를 수평으로 뒤집기
        transforms.ToTensor(),                         # 이미지를 텐서로 변환
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화
    ])

    # 지정된 경로에서 이미지 폴더 데이터셋 생성
    image_dataset = datasets.ImageFolder(data_dir, data_transform)

    # 훈련 데이터셋의 90%를 사용
    train_split = 0.9
    split_size = int(len(image_dataset) * train_split)
    batch_size = 64
    num_workers = 6

    # 데이터셋을 훈련 및 검증 세트로 분할
    train_set, valid_set = torch.utils.data.random_split(image_dataset, [split_size, len(image_dataset) - split_size])

    # 훈련 및 검증 세트를 로드하기 위한 데이터 로더 정의
    tr_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 데이터 로더를 사전에 저장
    dataloaders = {'train': tr_loader, 'val': val_loader}

    # 각 데이터셋의 크기를 사전에 저장
    dataset_sizes = {'train': split_size, 'val': len(image_dataset) - split_size}

    # 클래스 이름을 변수에 저장
    class_names = image_dataset.classes

    # 장치 설정 (GPU 사용 가능하면 사용, 그렇지 않으면 CPU 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
        since = time.time()  # 훈련 시작 시간 기록

        best_model_wts = copy.deepcopy(model.state_dict())  # 최적의 모델 가중치 저장을 위한 초기화
        best_acc = 0.0  # 최고 정확도 초기화

        for epoch in range(num_epochs):  # 각 에폭에 대한 반복
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:  # 훈련 및 검증 단계를 위한 반복
                if phase == 'train':
                    model.train()  # 모델을 훈련 모드로 설정
                else:
                    model.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0  # 현재 에폭의 손실 누적
                running_corrects = 0  # 현재 에폭의 정확한 예측 수

                for inputs, labels in dataloaders[phase]:  # 데이터 로더를 통해 배치 단위로 처리
                    inputs = inputs.to(device)  # 입력을 현재 디바이스(GPU/CPU)로 이동
                    labels = labels.to(device)  # 레이블을 현재 디바이스로 이동

                    optimizer.zero_grad()  # 그래디언트 초기화

                    # 그래디언트 계산 활성/비활성화 (훈련 단계에서만 활성화)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)  # 모델을 통해 출력 얻기
                        _, preds = torch.max(outputs, 1)  # 예측 클래스 얻기
                        loss = criterion(outputs, labels)  # 손실 계산

                        if phase == 'train':
                            loss.backward()  # 그래디언트 계산
                            optimizer.step()  # 모델 가중치 업데이트

                    running_loss += loss.item() * inputs.size(0)  # 전체 손실 누적
                    running_corrects += torch.sum(preds == labels.data)  # 정확한 예측 수 누적

                if phase == 'train':
                    scheduler.step()  # 스케줄러를 통해 학습률 업데이트

                epoch_loss = running_loss / dataset_sizes[phase]  # 에폭 손실 평균 계산
                epoch_acc = running_corrects.double() / dataset_sizes[phase]  # 에폭 정확도 계산

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  # 현재 에폭 손실 및 정확도 출력

                # 최고 검증 정확도 모델 저장
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since  # 전체 훈련 시간 계산
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(best_model_wts)  # 최고 성능 모델 가중치 로드
        return model  # 훈련된 모델 반환

    # ===========

    model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 사전 훈련된 ResNet-18 모델 로드

    num_ftrs = model_ft.fc.in_features  # 마지막 전결합층의 입력 특징 수 가져오기
    model_ft.fc = nn.Linear(num_ftrs, len(image_dataset.classes))  # 마지막 층을 데이터셋 클래스 수에 맞게 변경

    model_ft = model_ft.to(device)  # 모델을 현재 디바이스(GPU/CPU)로 이동

    criterion = nn.CrossEntropyLoss()  # 크로스 엔트로피 손실 함수 정의 (분류 작업용)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  # SGD 옵티마이저 정의, 학습률 0.001, 모멘텀 0.9
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 7 에폭마다 학습률을 0.1배 감소시키는 스케줄러 정의

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)  # 모델 훈련 시작, 총 10 에폭

    # 모델 가중치 저장
    torch.save(model_ft.state_dict(), 'D:\\AiHub\\model_ft.pth')


if __name__ == '__main__':
    main()

