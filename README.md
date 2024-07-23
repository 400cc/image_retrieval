# image_searching

도커 없이 로컬에서 테스트 한 점 참고해주시면 되겠습니다. 

<br>

## 테스트 환경
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1 \
GPU: NVIDIA GeForce RTX 3060 \
PostgreSQL: 13.15 \
pgAdmin4 : v8.8



<br>

## Installation
python은 3.8버전 이상, pytorch는 1.7 이상, torchvision은 0.8버전 이상이 필요합니다.

Grounded SAM을 사용하기 위해 GroundingDINO와 Segment Anything을 모두 설치해야 합니다.
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
git clone https://github.com/facebookresearch/segment-anything.git
```

<br>

현재 디렉토리에 clone 된 GroundingDINO, segment-anything에서 필요한 파일을 설치합니다.
```
python -m pip install -e segment-anything
python -m pip install -e GroundingDINO
```

<br>

이 때, visual studio 2019 'C++를 사용한 데스크톱 개발'이 설치돼있어야 GroundingDINO 의 dependency들을 설치할 수 있습니다. (만약 2022 버전에서 설치가 안되면 2019 버전으로 재설치해서 시도하면 될 것 같습니다.)

![alt text](assets/image.png)

<br>

Segment Anything의 pretrained weights를 다운받습니다. (encoder: vit-h)

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

<br> <br>
## 이미지 검색 Test
### database 연결
다음과 같이 API를 띄울 환경에 MySQL, PostgreSQL을 연결합니다.

* MySQL (MobaXterm에서 Tunneling)<br>
![image](https://github.com/user-attachments/assets/d354d887-c2da-4c27-bb48-0a66f84bd2a6)
* PostgreSQL (pgAdmin에서 Tunneling)<br>
![image](https://github.com/user-attachments/assets/9722764f-8492-4572-a827-d01f2c283c76)
![image](https://github.com/user-attachments/assets/6736a010-d5d8-4c23-8e30-6f38dc522b77)


### GroundingDINO 수정
GroundingDINO\groundingdino\util\inference.py 파일에 다음과 같은 함수를 추가해줍니다.
```python
def load_image_from_memory(image_source: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    # 이미지를 numpy 배열로 변환
    image = np.asarray(image_source)
    
    # 변환된 이미지를 텐서로 변환
    image_transformed, _ = transform(image_source, None)
    
    return image, image_transformed
```

### API 실행
다음과 같은 코드로 FastAPI를 실행합니다.
```
uvicorn image_search_api:app
```
