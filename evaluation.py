from __future__ import print_function, division

import pandas as pd
import torch
import torch.utils as utils
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm

from testDataset import TestDataset


def main():

    # Test 파일 다운받기
    # https://kr.object.ncloudstorage.com/aihub-competition/dataset/K-Fashion_Test.zip

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

    # 이미지 전처리를 위한 변환을 정의합니다. 순서대로 PIL 이미지로 변환, 크기 조절, 텐서로 변환, 정규화를 수행합니다.
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TestDataset 클래스의 인스턴스를 생성하여 테스트 데이터셋을 로드하고, 위에서 정의한 변환을 적용합니다.
    test_dataset = TestDataset('D:\\AiHub\\K-Fashion_Test\\', transform=test_transform)

    # 데이터 로더를 생성하여 배치 처리를 수행할 수 있게 합니다. batch_size는 256, 작업자 수는 6으로 설정합니다.
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=256, num_workers=6)

    # CUDA가 사용 가능한 경우 GPU를 사용하고, 그렇지 않은 경우 CPU를 사용하도록 설정합니다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 결과를 저장할 빈 리스트를 초기화합니다.
    result = []

    # 지정된 경로에서 이미지 폴더 데이터셋 생성
    image_dataset = datasets.ImageFolder(data_dir, data_transform)

    # 모델 구조 정의
    model_ft = models.resnet18() # ResNet-18 모델 구조

    # 클래스 수에 맞게 출력 레이어 조정 (이전 코드와 동일한 클래스 수를 지정해야 함)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(image_dataset.classes)) # len(image_dataset.classes)는 원래 데이터셋의 클래스 수와 일치해야 함

    try:
        # 1. GPU 사용 시
        # model_ft.load_state_dict(torch.load('model_ft.pth'))

        # 2. GPU 사용 불가 시
        model_ft.load_state_dict(torch.load('model_ft.pth', map_location=torch.device('cpu')))
    except Exception as e:
        print('Error:', e)

    # 평가 모드로 모델 설정
    model_ft.eval()

    # tqdm을 사용하여 진행 상황을 표시하면서 테스트 데이터 로더의 각 배치에 대해 반복합니다.
    for fnames, data in tqdm(test_dataloader):
        data = data.to(device) # 데이터를 GPU 또는 CPU로 이동합니다.
        output = model_ft(data) # 모델을 사용하여 입력 데이터에 대한 예측을 수행합니다.
        _, pred = torch.max(output, 1) # 예측 결과 중 가장 높은 값의 인덱스를 찾습니다.
        for j in range(len(fnames)): # 각 예측에 대해 반복합니다.
            result.append( # 결과 리스트에 예측 결과를 추가합니다.
                {
                    'filename': fnames[j],
                    'style': pred.cpu().detach().numpy()[j]
                }
            )

    # 결과를 파일 이름 순서대로 정렬하고, CSV 파일로 저장합니다.
    pd.DataFrame(sorted(result, key=lambda x: x['filename'])).to_csv('fashion_submission.csv', index=None)


if __name__ == '__main__':
    main()