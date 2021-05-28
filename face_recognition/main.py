import face_recognition
import os
from pymongo import MongoClient
from typing import List


def compute_folder(path: str) -> List[dict]:
    os.chdir(path)
    images_list = os.listdir()
    embeddings = []
    for image_path in images_list:
        image = face_recognition.load_image_file(image_path)
        emb_list = face_recognition.face_encodings(image)
        if len(emb_list) > 0:
            embeddings.append({'name': image_path, 'emb': emb_list[0].tolist()})
    os.chdir('../')
    return embeddings


def main(data_path: str):
    data = compute_folder(data_path)
    password = ''
    user = ''
    uri = f'mongodb://{user}:{password}@mongodb'

    client = MongoClient(uri)
    db = client['persons']
    collection = db['embeddings']
    x = collection.insert_many(data)
    print(x.inserted_ids)


if __name__ == '__main__':
    data_path = '/face_data'
    main(data_path)

