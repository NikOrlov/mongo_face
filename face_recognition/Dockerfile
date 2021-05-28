FROM animcogn/face_recognition

RUN pip install pymongo
COPY main.py /main.py
RUN mkdir /face_data
CMD ["python", "main.py"]
#CMD /usr/local/bin/shell.sh ; sleep infinity

