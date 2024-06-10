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
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```

<br>

이 때, visual studio 2019 'C++를 사용한 데스크톱 개발'이 설치돼있어야 GroundingDINO 의 dependency들을 설치할 수 있습니다. (2022버전으로는 잘 설치가 되지 않았습니다..)

![alt text](image.png)

<br>

Segment Anything의 pretrained weights를 다운받습니다. (encoder: vit-h)

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

<br> <br>

## pgvector

vector extension을 설치합니다.

```
CREATE EXTENSION vector;
```

<br>

임시 TABLE을 생성합니다. embedding vector 컬럼의 크기는 CLIP embedding 결과 길이가 512이므로 알맞게 설정합니다. 

```
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(512));
```

<br>

## Run segment_and_insert_pgvector.ipynb
cdn에 저장된 이미지들을 Grounded SAM으로 의상만 분할 후 CLIP encoder로 임베딩하여 pgvector에 저장하는 임시 파이프라인 코드입니다. 아직 cdn에 수집된 이미지가 없기 때문에 로컬에 저장된 이미지로 실행했습니다. 추후 cdn에서 이미지를 불러오고, 해당 이미지의 카테고리도 함께 가져오도록 수정되어야 합니다.

![alt text](image-1.png)

<br>

## Run test_similarity.ipynb

사용자에게 이미지와 카테고리를 입력받아 Grounded SAM으로 분할 후 CLIP encoder로 임베딩 후, pgvector에 저장된 이미지 임베딩들과 가장 높은 cosine similarity를 갖는 상위 5개 벡터를 보여주는 임시 코드입니다.

![alt text](image-3.png)

### output
```
Top similar images:
Image ID: 1, Similarity Score: 3.8470888600529562
Image ID: 5, Similarity Score: 4.106290058190262
Image ID: 4, Similarity Score: 6.348187477771091
Image ID: 2, Similarity Score: 6.4871681907635095
Image ID: 3, Similarity Score: 7.825285436524724
```
위 5개 이미지 중 같은 상품이면서 다른 이미지였던 첫 번째 이미지의 유사도가 가장 높게 나온 것을 확인할 수 있습니다.