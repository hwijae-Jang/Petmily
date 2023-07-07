import streamlit as st
import os
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image as img_prep
import numpy as np
import PIL
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

CLASS_NAMES1 = [
    "구진-플라크",
    "비듬각질상피성잔고리",
    "태선화-과다색소침착",
    "농포-여드름",
    "미란-궤양",
    "결절-종괴",
]

CLASS_NAMES2 = [
    "결막염",
    "궤양성각막질환",
    "백내장",
    "비궤양성각막질환",
    "색소침착성각막염",
    "안검내반증",
    "안검염",
    "안검종양",
    "유루증",
    "핵경화"
]

base_model = EfficientNetB0(weights="imagenet", include_top=False)
new_base_model = Model(base_model.inputs, base_model.layers[-1].output)
x = GlobalAveragePooling2D()(new_base_model.output)

# For Page 1 Model
new_output_layer1 = Dense(len(CLASS_NAMES1), activation='softmax')(x)
model1 = Model(new_base_model.inputs, new_output_layer1)

# For Page 2 Model
new_output_layer2 = Dense(len(CLASS_NAMES2), activation='softmax')(x)
model2 = Model(new_base_model.inputs, new_output_layer2)

def get_predictions(img, model):
    img_array = img_prep.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    
    return preds[0]

def get_sorted_predictions(img, model):
    preds = get_predictions(img, model)
    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)

    return sorted_preds

st.set_page_config(
    page_icon='😻🐶',
    page_title="Welcome to Petmily's website",
    layout='wide',
)

def main():
    st.title("Petmily ")
    menu = ["홈", "🐕피부질환", "🐕안구질환", "질병 종류"]
    choice = st.sidebar.selectbox("메뉴를 선택하세요", menu)

    if choice == "홈":
        st.subheader("저기요,, 정신이 좀 드세요?")
        st.write("")
        image_url = "https://onimg.nate.com/orgImg/sg/2017/08/02/20170801000436_0.jpg"  # 여기에 이미지 URL을 입력하세요.
        st.image(image_url, "Displayed Image", use_column_width=True)
    
    elif choice == "🐕피부질환":
        st.subheader("반려동물 피부질환 예측 모델")
        uploaded_file = st.file_uploader("이미지를 선택해주세요...",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            resized_image = input_image.resize((224, 224), PIL.Image.ANTIALIAS)
            st.image(input_image, "Uploaded Image", use_column_width=True)

            sorted_predictions = get_sorted_predictions(resized_image, model1)

            st.write("질병별 예측 확률:")
            for i, (index, pred) in enumerate(sorted_predictions):
                class_name = CLASS_NAMES1[index]
                st.write(f"{i + 1}. {class_name}: {pred * 100:.2f}%")
    
    elif choice == "🐕안구질환":
        st.subheader("반려동물 안구질환 예측 모델")
        uploaded_file = st.file_uploader("이미지를 선택해주세요...",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            resized_image = input_image.resize((224, 224), PIL.Image.ANTIALIAS)
            st.image(input_image, "Uploaded Image", use_column_width=True)

            sorted_predictions = get_sorted_predictions(resized_image, model2)

            st.write("질병별 예측 확률:")
            for i, (index, pred) in enumerate(sorted_predictions):
                class_name = CLASS_NAMES2[index]
                st.write(f"{i + 1}. {class_name}: {pred * 100:.2f}%")
                
    elif choice == '질병 종류':
         # 이미지 url 및 설명을 포함한 데이터 프레임 생성
        data = {
        "병변": ["반점(macule)", "반(pathces)", "구진(papule)", "판(plaque)", "농포(pustule)", 
                "소수포(vescle)", "대수포(bulla)", "팽진(wheal)", "결절(nodule)", "종양(tumor)", "낭포(cyst)"],
        "이미지": ["https://mediahub.seoul.go.kr/uploads/2013/04/2013040506163065_mainimg.jpg",
                "https://example.com/path/to/image2.jpg",
                "https://example.com/path/to/image3.jpg",
                "https://example.com/path/to/image4.jpg",
                "https://example.com/path/to/image5.jpg",
                "https://example.com/path/to/image6.jpg",
                "https://example.com/path/to/image7.jpg",
                "https://example.com/path/to/image8.jpg",
                "https://example.com/path/to/image9.jpg",
                "https://example.com/path/to/image10.jpg",
                "https://example.com/path/to/image11.jpg"],
        "설명": ["1cm 이상 피부색이 변화된 점",
                "1cm 미만 피부색이 변화된 점", 
                "1cm 이하 작고, 돌출된 단단한 병변", 
                "1cm 보다 큰 평평한 돌출된 단단한 병변", 
                "농이 찬 병변", 
                "1cm 이하 혈청이 찬 병변", 
                "1cm 보다 큰 수포", 
                "부종으로 구성된 경계가 명확한 돌출된 병변", 
                "경계가 명확한 직경 1cm이상의 단단한 육기부로 보통 피부의 심층부로 확대됨", 
                "피부나 피하조직의 모든 구조를 포함하는 큰 덩어리", 
                "막으로 둘러싼 액체나 반고체 물질을 갖는 병변"]
        }
        df = pd.DataFrame(data)

        # 데이터 프레임의 이미지 url을 실제 이미지로 변환하는 함수
        def image_formatter(image_url):
            return f'<img src="{image_url}" width="125" length="125" />'

        # 이미지와 설명의 좌우 정렬을 설정하는 함수
        def align_formatter(text, align="left"):
            return f'<p style="text-align: {align};">{text}</p>'

        # 데이터 프레임의 각 열에 서식 적용
        columns = ["병변", "사진", "설명"]
        df_styled = df.style.format({
            "병변": lambda text: align_formatter(text, "center"),
            "사진": image_formatter,
            "설명": lambda text: align_formatter(text, "justify")
        })

        # 데이터 프레임 출력
        st.write(df_styled.to_html(escape=False), unsafe_allow_html=True)
                
if __name__ == '__main__':
    main()
