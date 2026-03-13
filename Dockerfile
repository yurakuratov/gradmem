FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37

WORKDIR /home/jovyan/test-time-gd

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
