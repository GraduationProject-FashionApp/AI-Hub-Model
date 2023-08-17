import os

import numpy as np
import torch.utils as utils
from PIL import Image


# 사용자 정의 데이터셋 클래스를 정의합니다. PyTorch의 Dataset 클래스를 상속받아 사용합니다.
class TestDataset(utils.data.Dataset):
    # 생성자 메서드로, 객체 초기화 시에 호출됩니다.
    def __init__(self, root, transform=None):
        self.root = root  # 이미지 파일의 루트 디렉토리 경로를 저장합니다.
        self.image_list = os.listdir(root)  # 루트 디렉토리에서 모든 이미지 파일의 이름을 리스트로 저장합니다.
        self.transform = transform  # 이미지 전처리를 수행할 변환을 저장합니다.

    # 데이터셋의 길이(이미지 개수)를 반환합니다.
    def __len__(self):
        return len(self.image_list)

    # 인덱스에 해당하는 이미지를 불러와 전처리를 수행하고, 이미지 파일의 이름과 함께 반환합니다.
    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_list[index])  # 해당 인덱스의 이미지 경로를 생성합니다.
        image = np.array(Image.open(image_path))  # 이미지를 불러와 numpy 배열로 변환합니다.
        image = self.transform(image)  # 지정된 변환을 이미지에 적용합니다.
        return self.image_list[index], image  # 이미지 파일의 이름과 변환된 이미지를 반환합니다.
