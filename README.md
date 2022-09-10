## File hướng dẫn chạy code nộp vòng private test của đội UIT_AI_Nanoparticles

### Docker:

Load docker image: `docker load -i submission.tar`
Run docker image: `docker run -it --name aqi_submit -v /app aqi_submit /bin/bash`

### Data:
- Folder chứa data dùng để train đặt trong folder "train" cùng folder với file này. Format folder như dữ liệu đã công bố:
```
README.md
train
|
\ air
| \ <csv>
|
\ meteo
  \ <csv>
...
```

- Folder chứa data dùng để test và tạo file đáp án được đặt trong folder "test" cùng folder với file này. Format folder tương tự như đã công bố:
```
README.md
test
|
\ input
  \ 1
  \ 2
  \ 3
  \ ...
...
```

### Chạy files:
- Cài đặt các thư viện cần thiết (không cài sẵn khi build docker vì lý do bộ nhớ):
```
pip install -r requirements.txt
```

- Chạy một lần:
    + __Để tiện việc chấm bài, cả 3 lệnh giới thiệu dưới đây đã được gói lại trong 1 file bash là `run.sh`, chạy lệnh `bash run.sh` để chạy cùng lúc cả 3 lệnh thay vì chạy từng lệnh một như ở bên dưới.__

- Chạy từng lệnh:
    - Chạy 2 file dùng để train model:
        + Model _forecaster_: `python3 forecaster.py`
        + Model _extrapolator_: `python3 extrapolator.py`
        + Hai file này sẽ sử dụng kha khá tài nguyên (3.5GB RAM, 2.7GB GPU), tổng thời gian train khoảng 1h-1h20ph nếu sử dụng GPU.

    - Sau khi chạy hai file trên, weights của hai model sẽ được lưu lại trong 2 file h5 trong folder `weights`, và tham số data của forecaster trong folder `paras`.

    - Chạy file cuối cùng để tạo file kết quả: `python3 e2e_model.py`