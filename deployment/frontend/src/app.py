import os
import streamlit as st
from random import random
from requests import post


class Frontend:
    def __init__(self):
        self.__draw_page()

    def __draw_page(self):
        st.title("Enter the comment:")

        self.text_input = st.text_area("", "", height=200)

        if self.text_input:
            self.__predict()

    def __make_request(self, request):
        backend_url = 'http://' + os.environ["BACKEND_URL"] + ":" + os.environ["BACKEND_PORT"]
        predictions = post(backend_url + "/predict", json={"input": request}).json()
        st.warning(predictions)
        return predictions["predictions"]

    def __predict(self):
        data = self.__make_request(self.text_input)
        stars = data.index(max(data)) + 1

        st.text("Rating: " + " ".join(["⭐"] * stars))
        st.bar_chart(data=data)


Frontend()
