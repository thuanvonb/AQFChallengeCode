<<<<<<< HEAD
## UIT_AI_Nanoparticles submission for Air Quality Forecasting Challenge
=======
## File hướng dẫn chạy code nộp vòng private test của đội UIT_AI_Nanoparticles

### Docker:

- Load docker image: `docker load -i submission.tar`
- Run docker image: `docker run -it --name aqi_submit -v /app aqi_submit /bin/bash`
>>>>>>> dd25e456cf5e9f564f6ce03c8e491e46b110a430

### Data:
- Folder chứa data dùng để train đặt trong folder "train" cùng folder với file này:
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

- Folder chứa data dùng để test và tạo file đáp án được đặt trong folder "test" cùng folder với file này:
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
- Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```

- Chạy một lần:
    + __Để thuận tiện, cả 3 lệnh giới thiệu dưới đây đã được gói lại trong 1 file bash là `run.sh` (linux) và `run.bat`, chạy các file này để chạy cùng lúc cả 3 lệnh thay vì chạy từng lệnh một như ở bên dưới.__

- Chạy từng lệnh:
    - Chạy 2 file dùng để train model:
        + Model _forecaster_: `python3 forecaster.py`
        + Model _extrapolator_: `python3 extrapolator.py`
        + Hai file này sẽ sử dụng kha khá tài nguyên, tổng thời gian train khá lâu (nếu không dùng GPU)

    - Sau khi chạy hai file trên, weights của hai model sẽ được lưu lại trong 2 file h5 trong folder `weights`, và tham số data của forecaster trong folder `paras`.

    - Chạy file cuối cùng để tạo file kết quả: `python3 e2e_model.py`
<<<<<<< HEAD
    
 ### Optional Flags:
 - Khi chạy từng lệnh thì bạn có thể tùy chỉnh một số biến số của mô hình và quá trình training bằng cách dùng các flags được cung cấp
 - Để biết mỗi file có những flags nào, chạy `python <tên file>.py --help` để list ra số flags và mô tả của chúng.
 
 
=======
>>>>>>> dd25e456cf5e9f564f6ce03c8e491e46b110a430
