# 학습된 모델로 실제 손글씨 인식시키기

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# 이미지 파일 불러와서 그레이스케일 이미지로 변환
img_color = cv.imread('bigsize_writing.png', cv.IMREAD_COLOR)
# img_color = cv.imread('writing.png', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

# 그레이스이미지 바이너리로 변환
ret, img_binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

# 모폴로지 연산 적용(이진화 결과 혹시 모를 빈 공간 메꾸기)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)

# cv.imshow('digit', img_binary)
# cv.waitKey(0)

# 숫자별로 분리하기 위해 컨투어 검출
contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # 숫자별로 경계박스 구하기
    x, y, w, h = cv.boundingRect(contour)

    # 가로,세로 중 긴 방향 선택, 여분추가해 한 변의 크기 정하기
    # 잘라낼 이미지를 저장할 빈 이미지 생성
    length = max(w, h) + 60
    img_digit = np.zeros((length, length, 1), np.uint8)

    # 숫자가 이미지의 정중앙에 오도록 경계박스의 시작 위치 조정
    new_x, new_y = x - (length - w) // 2, y - (length - h) // 2

    # 바이너리 이미지에서 숫자 영역 가져와 변수에 저장
    img_digit = img_binary[new_y:new_y + length, new_x:new_x + length]

    # 숫자가 잘 인식되도록 팽창 모폴로지 연산 적용
    kernel = np.ones((5, 5), np.uint8)
    img_digit = cv.morphologyEx(img_digit, cv.MORPH_DILATE, kernel)

    # cv.imshow('digit', img_digit)
    # cv.waitKey(0)

    # 학습된 모델 불러오기
    model = load_model("./model/otherperson_model.h5")

    # 이미지 크기를 학습된 모델에서 요구하는 가로세로 28로 변환
    img_digit = cv.resize(img_digit, (28, 28), interpolation=cv.INTER_AREA)

    img_digit = img_digit / 255.0  # 이미지 픽셀도 변환

    img_input = img_digit.reshape(1, 28, 28, 1)  # 이미지 형태도 변환

    # 이미지를 입력으로 사용해 예측 진행
    predictions = model.predict(img_input)

    # argmax함수를 사용해 softmax 결과를 숫자로 변환
    number = np.argmax(predictions)
    print(number)

    # 원본 이미지의 숫자마다 사각형 그려주기
    cv.rectangle(img_color, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 이미지에 있는 숫자 위에 인식된 숫자 적어주기
    location = (x + int(w * 0.5), y - 10)
    font = cv.FONT_HERSHEY_COMPLEX
    fontScale = 1.2
    cv.putText(img_color, str(number), location, font, fontScale, (0, 255, 0), 2)

    # 이미지에서 잘라낸 숫자를 가공한 결과
    cv.imshow('digit', img_digit)
    cv.waitKey(0)

# 원본 이미지에 인식한 숫자를 적은 결과
cv.imshow('result', img_color)
cv.waitKey(0)