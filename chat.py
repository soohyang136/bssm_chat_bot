import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

st.set_page_config(layout='wide')

st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage](https://school.busanedu.net/bssm-h/main.do) |
    [Instagram](https://www.instagram.com/bssm.hs/) |
    [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    tel : 051-971-2153
    """
)

tab1, tab2 = st.tabs(["학교소개","문의"])

with tab1:
    st.title("저희 소마고를 소개합니다")
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css('style.css')
    st.header("부산소프트웨어마이스터고등학교")
    st.text("설립구분:공립   설립유형:단설   학교특성:특수목적고등학교")
    st.text("설립일자 : 1970년 03월 26일")
    st.text("학생수 : 125명 (남 89명 , 여 36명, 성비:2.5:1)")
    st.text("교원수 : 33명 (남 13명 , 여 20명)")
    st.text("주소 :부산광역시 강서구 가락대로 1393")
    
    
with tab2:
    st.header("챗봇에게 무엇이든 물어보세요!")
    @st.cache(allow_output_mutation=True)
    def cached_model():
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        return model
    @st.cache(allow_output_mutation=True)
    def get_dataset():
        df = pd.read_csv('bsg_chat.csv')
        df['embedding'] = df['embedding'].apply(json.loads)
        return df
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css('style.css')

    model = cached_model()
    df = get_dataset()



    st.header('부산소프트웨어마이스터고 챗봇')
    st.subheader('안녕하세요 소마고 챗봇입니다.')

    with st.form('form', clear_on_submit=True):
        user_input = st.text_input('사용자 : ', '')
        submitted = st.form_submit_button('전송')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if submitted and user_input:
        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]

        st.session_state.past.append(user_input)
        if answer['distance'] > 0.5:
            st.session_state.generated.append(answer['챗봇'])
        else :
            st.session_state.generated.append("051-971-2153으로 연락하세요.")
    for i in range(len(st.session_state['past'])):
        st.markdown(
        """
        <div class="container">
            <div class="leftBox">
                <p>유저</p>
                <p class="text">{0}</p>
            </div>
            <div class="rightBox">
                <p>챗봇</p>
                <p class="text">{1}</p>
            </div<
        <div/>
        """
        .format(st.session_state['past'][i], st.session_state['generated'][i]), unsafe_allow_html=True
        )




                                 

