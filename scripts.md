# Project Runbook

Tất cả lệnh dưới đây được chạy từ thư mục gốc của project:

```bash
cd /media/lovecrush/LoveCrush/Documents/project/pro_Yelp_Hybrid_RecSys_FinUniPro
```

## 1. Chạy API local

```bash
uvicorn api.main:app --reload
```

- `uvicorn`: ASGI server để chạy FastAPI app.
- `api.main:app`: nghĩa là import object `app` từ file `api/main.py`.
- `--reload`: tự restart server khi code thay đổi, chỉ nên dùng khi development local.

Khi server chạy:

- `http://127.0.0.1:8000/basic`: basic demo + explainability + map.
- `http://127.0.0.1:8000/experience`: UX login/feed page.
- `http://127.0.0.1:8000/management`: management dashboard.
- `http://127.0.0.1:8000/docs`: FastAPI Swagger docs.

## 2. Preprocess Yelp raw data

Mặc định hiện tại, pipeline có thể chạy theo kiểu `auto`:

- nếu `data/business_main.parquet` và `data/reviews.parquet` tồn tại, preprocess sẽ ưu tiên đọc parquet
- nếu không có parquet export, script sẽ fallback về `raw_data/*.json`

### 2.1. Chạy nhanh với một phần dữ liệu

```bash
python3 scripts/preprocess_yelp.py \
  --input-format auto \
  --data-dir data \
  --out-dir processed \
  --usa-only \
  --max-review-chunks 8 \
  --max-aux-chunks 8 \
  --progress-every 5
```

- Dùng khi muốn test pipeline nhanh.
- `--input-format auto`: tự chọn `parquet` nếu có, nếu không thì dùng `json`.
- `--data-dir data`: thư mục chứa parquet export.
- `--max-review-chunks 8`: chỉ đọc 8 chunk review đầu tiên.
- `--max-aux-chunks 8`: chỉ đọc 8 chunk của tip/checkin.
- `--usa-only`: chỉ giữ restaurant thuộc các bang của US.
- Output chính:
  - `processed/businesses.pkl`
  - `processed/interactions.pkl`
  - `processed/preprocess_summary.json`

### 2.2. Chạy toàn bộ dữ liệu US restaurants

```bash
python3 scripts/preprocess_yelp.py \
  --input-format parquet \
  --data-dir data \
  --out-dir processed \
  --usa-only \
  --max-review-chunks 0 \
  --max-aux-chunks 0 \
  --progress-every 5
```

- `0` nghĩa là không giới hạn phần dữ liệu được đọc từ source đã chọn.
- Đây là chế độ phù hợp khi build artifact thật để demo/presentation.
- Khi chọn `--input-format parquet`, script sẽ lấy trực tiếp từ:
  - `data/business_main.parquet`
  - `data/business_attributes.parquet`
  - `data/business_soup.parquet`
  - `data/business_popularity.parquet`
  - `data/reviews.parquet`

### 2.3. Một số tham số quan trọng

```bash
python3 scripts/preprocess_yelp.py \
  --out-dir processed \
  --usa-only \
  --min-business-reviews 20 \
  --min-user-reviews 3 \
  --min-item-reviews 5 \
  --chunksize 200000
```

- `--min-business-reviews 20`: business phải có ít nhất 20 review mới được giữ.
- `--min-user-reviews 3`: user phải có ít nhất 3 interaction để vào tập train.
- `--min-item-reviews 5`: business phải có ít nhất 5 interaction sau bước build interactions.
- `--chunksize 200000`: số dòng JSON được đọc mỗi chunk.

## 3. Train hybrid recommender

### 3.1. Train từ Yelp-only data

```bash
python3 scripts/train_recommender.py \
  --processed-dir processed \
  --artifact-dir artifacts \
  --local-data-dir local_data \
  --dataset-mode yelp_only \
  --max-eval-users 300 \
  --progress-every 50
```

- `--dataset-mode yelp_only`: chỉ train bằng Yelp dataset đã preprocess.
- `--max-eval-users 300`: evaluate offline trên tối đa 300 users để nhanh hơn.
- `--progress-every 50`: in tiến độ evaluate mỗi 50 users.

### 3.2. Train bằng Yelp + local user interactions

```bash
python3 scripts/train_recommender.py \
  --processed-dir processed \
  --artifact-dir artifacts \
  --local-data-dir local_data \
  --dataset-mode merged \
  --max-eval-users 300 \
  --progress-every 50
```

- `--dataset-mode merged`: gộp `processed/interactions.pkl` với `local_data/interactions.jsonl`.
- Dùng khi muốn re-train sau khi user local đã login/rate business trên demo.

### 3.3. Chỉ train bằng local users

```bash
python3 scripts/train_recommender.py \
  --processed-dir processed \
  --artifact-dir artifacts \
  --local-data-dir local_data \
  --dataset-mode local_only \
  --max-eval-users 100
```

- Chỉ hữu ích khi bạn muốn test luồng local user.
- Không phù hợp để tạo model chính vì dữ liệu local thường rất nhỏ.

### 3.4. Tham số model/evaluation quan trọng

```bash
python3 scripts/train_recommender.py \
  --processed-dir processed \
  --artifact-dir artifacts \
  --local-data-dir local_data \
  --dataset-mode yelp_only \
  --k 10 \
  --min-history 3 \
  --n-test-items 1 \
  --relevance-threshold 4.0 \
  --prior-weight 0.1 \
  --bundle-event-users 300 \
  --max-events-per-user 200
```

- `--k 10`: top-K dùng cho offline evaluation.
- `--min-history 3`: user cần tối thiểu 3 interaction trong train trước khi hold-out test item.
- `--n-test-items 1`: giữ lại 1 interaction cuối cùng của mỗi user để test.
- `--relevance-threshold 4.0`: chỉ các held-out item có rating từ 4 sao trở lên mới được coi là relevant trong offline ranking metrics.
- `--prior-weight 0.1`: trọng số popularity prior trong final score hybrid.
- `--bundle-event-users 300`: chỉ lưu activity logs của một tập user demo nhỏ trong bundle để giảm size artifact.
- `--max-events-per-user 200`: số sự kiện tối đa lưu cho mỗi user demo đó.

## 4. Chạy full pipeline end-to-end

### 4.1. Pipeline nhanh với partial data

```bash
python3 scripts/run_pipeline.py \
  --input-format auto \
  --data-dir data \
  --dataset-mode yelp_only \
  --max-review-chunks 8 \
  --max-aux-chunks 8 \
  --max-eval-users 200 \
  --progress-every 50
```

- Chạy theo thứ tự:
  1. preprocess
  2. train
  3. checkpoint

### 4.2. Pipeline full data

```bash
python3 scripts/run_pipeline.py \
  --input-format parquet \
  --data-dir data \
  --dataset-mode yelp_only \
  --full-data \
  --max-eval-users 300 \
  --progress-every 50
```

- `--full-data`: tự chuyển `max-review-chunks` và `max-aux-chunks` về `0`.
- Phù hợp khi bạn muốn build lại toàn bộ artifact từ đầu.

## 5. Full USA retrain shortcut

```bash
python3 scripts/run_full_usa_retrain.py \
  --input-format parquet \
  --data-dir data \
  --dataset-mode yelp_only \
  --max-eval-users 300 \
  --k 10 \
  --min-history 3 \
  --n-test-items 1 \
  --relevance-threshold 4.0 \
  --prior-weight 0.1 \
  --progress-every 50
```

- Đây là shortcut dành riêng cho full US restaurants.
- Script sẽ tự chạy:
  1. full preprocess
  2. train hybrid bundle
  3. deployment checkpoint

## 6. Checkpoint codebase/deploy

```bash
python3 scripts/checkpoint.py \
  --raw-data-dir raw_data \
  --processed-dir processed \
  --bundle-path artifacts/model_bundle.pkl
```

- Dùng để kiểm tra nhanh codebase hiện tại đang ở trạng thái `GOOD` hay `BAD`.
- Script sẽ check:
  - dependencies
  - raw files
  - processed files
  - bundle load
  - sample inference

## 7. Chạy API sau khi train xong

```bash
uvicorn api.main:app --reload
```

Sau đó kiểm tra:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/checkpoint
```

## 8. Gợi ý workflow chuẩn

### 8.1. Khi đang phát triển nhanh

```bash
python3 scripts/run_pipeline.py \
  --dataset-mode yelp_only \
  --max-review-chunks 8 \
  --max-aux-chunks 8 \
  --max-eval-users 200
uvicorn api.main:app --reload
```

### 8.2. Khi muốn build artifact thật để demo

```bash
python3 scripts/run_full_usa_retrain.py \
  --dataset-mode yelp_only \
  --max-eval-users 300 \
  --progress-every 50
uvicorn api.main:app --reload
```

### 8.3. Khi muốn re-train với local users đã tương tác

```bash
python3 scripts/train_recommender.py \
  --processed-dir processed \
  --artifact-dir artifacts \
  --local-data-dir local_data \
  --dataset-mode merged \
  --max-eval-users 300
uvicorn api.main:app --reload
```

## 9. Diễn giải ngắn gọn các khái niệm lệnh

- `processed-dir`: nơi chứa bảng đã preprocess.
- `artifact-dir`: nơi chứa model artifact cuối cùng, quan trọng nhất là `model_bundle.pkl`.
- `local-data-dir`: nơi chứa local users và interactions của demo app.
- `dataset-mode yelp_only`: chỉ dùng dữ liệu Yelp.
- `dataset-mode merged`: Yelp + local user data.
- `dataset-mode local_only`: chỉ local data.
- `progress-every`: bao lâu in tiến độ một lần.
- `quiet`: giảm log terminal.

## 10. File chính liên quan tới các lệnh

- `scripts/preprocess_yelp.py`: preprocess raw Yelp JSON.
- `scripts/train_recommender.py`: fit model, evaluate, save bundle.
- `scripts/run_pipeline.py`: orchestration nhanh.
- `scripts/run_full_usa_retrain.py`: orchestration full US data.
- `scripts/checkpoint.py`: health classification cho terminal/deploy.
- `api/main.py`: FastAPI app entrypoint cho `uvicorn api.main:app --reload`.
