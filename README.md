# 🎬 OTT Recommendation System

Hệ thống gợi ý nội dung đa thuật toán cho nền tảng OTT (Over-The-Top). Đọc dữ liệu tương tác người dùng từ PostgreSQL và tạo các gợi ý phim/series cá nhân hoá.

---

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Seed dữ liệu test](#-seed-dữ-liệu-test)
- [Sử dụng CLI](#-sử-dụng-cli)
- [Sử dụng API](#-sử-dụng-api)
- [Pipeline ML nâng cao](#-pipeline-ml-nâng-cao)
- [Signal Weight → Rating](#-signal-weight--rating)
- [Đánh giá mô hình](#-đánh-giá-mô-hình)
- [Triển khai](#-triển-khai)

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| **Multi-Stage Pipeline** | Pipeline ML đa giai đoạn: Candidate Gen → Feature Eng → LightGBM Ranking |
| **Model Persistence** | Lưu/tải model đã train với `joblib` |
| **EDA** | Trực quan hoá: phân bố rating, hoạt động user, sparsity heatmap |
| **CLI** | Giao diện dòng lệnh đầy đủ qua `typer` |
| **REST API** | FastAPI phục vụ model LightGBM với độ trễ thấp |
| **Seed Data** | Script seed dữ liệu test đầy đủ cho cả 2 database |
| **Seed Data** | Script seed dữ liệu test đầy đủ cho cả 2 database |

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Audit DB    │────▶│  Recommendation      │◀────│  Content DB        │
│  (audit_logs)│     │  System (Python)     │     │  (content, movies, │
└──────────────┘     │                      │     │   tvseries, ...)   │
                     │  ┌──────────────┐    │     └────────────────────┘
                     │  │ ML Pipeline  │    │
                     │  │ (LightGBM)   │    │
                     │  └──────────────┘    │
                     │         │            │
                     │    ┌────▼────┐       │
                     │    │ FastAPI │       │
                     │    │  / CLI  │       │
                     │    └─────────┘       │
                     └──────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
recommendation_system/
├── api.py                      # FastAPI REST API
├── cli.py                      # Typer CLI interface
├── eda.py                      # EDA + visualizations
├── requirements.txt            # Dependencies
├── .env                        # Biến môi trường (không commit)
├── .env.example                # Template biến môi trường
│
├── config/
│   └── setting.py              # Cấu hình: DB URLs, hyperparameters
│
├── data/
│   ├── db.py                   # SQLAlchemy engines (audit + content)
│   ├── load_auditlog.py        # Load dữ liệu từ bảng audit_logs
│   ├── load_content.py         # Load dữ liệu từ bảng content + relations
│   ├── preprocessing.py        # Làm sạch, rating matrix, báo cáo chất lượng
│   ├── seed_data.py            # 🌱 Script seed dữ liệu test
│   └── saved_models/           # Model đã lưu
│
├── models/
│   └── persistence.py          # Lưu/tải model
│
├── pipeline/
│   ├── features.py             # Feature engineering cho pipeline
│   ├── train.py                # Training pipeline đa giai đoạn
│   └── serve.py                # Serving predictions
│
└── utils/
    ├── logger.py               # Logging tiện ích
    └── timer.py                # Timer decorator
```

---

## 💻 Yêu cầu hệ thống

- **Python** 3.10+
- **PostgreSQL** 14+ (2 database riêng biệt: audit-service-db và content-service-db)
- **pip** hoặc **virtualenv**

---

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone <repo-url>
cd CINEMAKATOK26-RECOMMENDATION-SYSTEM
```

### Bước 2: Tạo virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Nếu gặp lỗi với `psycopg2-binary`, hãy cài đặt PostgreSQL development headers hoặc thử `pip install psycopg2`.

### Bước 4: Cấu hình database

Copy file `.env.example` thành `.env` và điền thông tin kết nối:

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# URL kết nối PostgreSQL cho Audit Database
POSTGRES_URL_AUDIT=postgresql://postgres:your_password@localhost:5432/audit_service_db

# URL kết nối PostgreSQL cho Content Database
POSTGRES_URL_CONTENT=postgresql://postgres:your_password@localhost:5432/content_service_db
```

> ⚠️ **Quan trọng:** Hệ thống cần **2 database riêng biệt**:
> - `audit_service_db` — lưu trữ bảng `audit_logs` (dữ liệu tương tác user)
> - `content_service_db` — lưu trữ các bảng `content`, `movies`, `tvseries`, `season`, `episode`, `actor`, `director`, `category`, `tag` và các bảng junction

### Bước 5: Tạo database (nếu chưa có)

```sql
-- Chạy trong psql hoặc pgAdmin
CREATE DATABASE audit_service_db;
CREATE DATABASE content_service_db;
```

---

## 🌱 Seed dữ liệu test

Script seed sẽ tạo dữ liệu mẫu cho cả 2 database để test hệ thống.

### Dữ liệu được tạo

| Database | Bảng | Số lượng |
|----------|------|----------|
| Content DB | `content` | 50 (25 phim + 25 series) |
| Content DB | `movies` | 25 (kèm duration) |
| Content DB | `tvseries` | 25 |
| Content DB | `season` | 25 (1 season/series) |
| Content DB | `episode` | ~180 episodes |
| Content DB | `category` | 15 thể loại |
| Content DB | `tag` | 20 tags |
| Content DB | `actor` | 30 diễn viên |
| Content DB | `director` | 15 đạo diễn |
| Audit DB | `audit_logs` | ~500 lượt tương tác |
| Audit DB | Users (unique) | 30 user ảo |

### Chạy seed

```bash
# Seed cả 2 database (drop tables cũ & tạo mới)
python -m data.seed_data --clean

# Chỉ seed Content DB
python -m data.seed_data --only content --clean

# Chỉ seed Audit DB (cần Content DB có dữ liệu trước)
python -m data.seed_data --only audit --clean

# Seed thêm dữ liệu (không xoá dữ liệu cũ) — sẽ bỏ qua nếu đã có dữ liệu
python -m data.seed_data
```

> **Lưu ý:** Seed audit **phải chạy sau** content vì audit_logs tham chiếu đến movie/series IDs.

### Kết quả kỳ vọng

```
============================================================
🌱 RECOMMENDATION SYSTEM — DATA SEEDER
============================================================
🎬 Seeding Content DB...
  📦 Content tables created
  ✅ Inserted 15 categories
  ✅ Inserted 20 tags
  ✅ Inserted 30 actors
  ✅ Inserted 15 directors
  ✅ Inserted 25 movies
  ✅ Inserted 25 TV series with seasons & episodes
  ✅ Total content items: 50

📊 Seeding Audit DB...
  ✅ Inserted 502 audit log entries
  👤 Users: 30

============================================================
✅ Seeding complete!
============================================================
```

---

## 🖥️ Sử dụng CLI

### Train model

```bash
# Train Multi-Stage ML Pipeline (LightGBM)
python cli.py train
```

### Lấy recommendations

```bash
# Lấy danh sách gợi ý cho user (thay <USER_ID> bằng UUID thực tế)
python cli.py recommend --user-id <USER_ID> --top-n 10
```

> 💡 **Mẹo:** Để lấy danh sách user ID, truy vấn trực tiếp database:
> ```sql
> SELECT DISTINCT "userId" FROM audit_logs LIMIT 5;
> ```

### EDA (Phân tích dữ liệu)

```bash
# Tạo biểu đồ EDA
python cli.py eda
# → Output: data/plots/
```

### Xem model đã lưu

```bash
python cli.py models
```

---

## 🌐 Sử dụng API

### Khởi động server

```bash
# Cách 1: uvicorn (khuyên dùng khi dev)
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Cách 2: chạy trực tiếp
python api.py
```

Sau khi server chạy, truy cập: **http://localhost:8000/docs** để xem Swagger UI tương tác.

### Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| `GET` | `/health` | Kiểm tra trạng thái server |
| `GET` | `/recommend/{user_id}?top_n=10` | Gợi ý phim qua LightGBM Pipeline |
| `GET` | `/models` | Danh sách model đã lưu |

### Ví dụ gọi API

```bash
# Lấy gợi ý
curl "http://localhost:8000/recommend/<USER_ID>?top_n=10"
```

### Response mẫu

```json
{
  "user_id": "a1b2c3d4-...",
  "model": "multi_stage_pipeline",
  "recommendations": [
    {
      "itemid": "uuid-of-movie",
      "lgb_score": 0.85,
      "title": "Oppenheimer",
      "type": "MOVIE"
    },
    {
      "itemid": "uuid-of-series",
      "lgb_score": 0.76,
      "title": "Squid Game Season 2",
      "type": "TVSERIES"
    }
  ]
}
```

---

---

## 📊 Signal Weight → Rating

Hệ thống chuyển đổi `signalWeight` từ audit_logs sang rating để train model:

| signalWeight | Ý nghĩa | Hành động | Rating |
|---|---|---|---|
| `2` | Tín hiệu mạnh | `PLAY_MOVIE`, `PLAY_EPISODE_OF_SERIES`, `LIKE_MOVIE`, `LIKE_SERIES` | `5` |
| `1` | Tín hiệu trung bình | `ADD_MOVIE_TO_WATCHLIST`, `ADD_SERIES_TO_WATCHLIST` | `3` |
| `-1` | Tín hiệu tiêu cực | `UNLIKE_MOVIE`, `UNLIKE_SERIES`, `REMOVE_*_FROM_WATCHLIST` | `1` (loại mặc định) |
| `0` | Bỏ qua | Các action admin/system | Loại bỏ |

---

## 📈 Đánh giá mô hình

Các metrics được sử dụng:

| Metric | Mô tả |
|---|---|
| **RMSE** | Root Mean Squared Error — càng thấp càng tốt |
| **MAE** | Mean Absolute Error — sai số trung bình |
| **Precision@K** | Tỷ lệ item gợi ý thực sự liên quan |
| **Recall@K** | Tỷ lệ item liên quan được gợi ý |
| **Hit Rate** | Tỷ lệ user có ít nhất 1 gợi ý đúng |
| **Coverage** | Phần trăm item có thể gợi ý |

---

## 🚢 Triển khai

### Triển khai trên Render

1. Thiết lập environment variables trên Render dashboard:
   ```
   POSTGRES_URL_AUDIT=postgresql://...
   POSTGRES_URL_CONTENT=postgresql://...
   APP_URL=https://your-app-name.onrender.com
   ```

2. Build command:
   ```bash
   pip install -r requirements.txt
   ```

3. Start command:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

> **Lưu ý:** API tự động reload dữ liệu từ DB mỗi 15 phút và ping `/health` mỗi 14 phút để tránh sleep trên free tier.

---

## 🔧 Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `connection refused` | PostgreSQL chưa chạy hoặc sai port | Kiểm tra PostgreSQL đang chạy và port đúng trong `.env` |
| `database does not exist` | Chưa tạo database | Tạo DB: `CREATE DATABASE audit_service_db;` |
| `relation does not exist` | Chưa seed data hoặc bảng chưa tạo | Chạy: `python -m data.seed_data --clean` |
| `POSTGRES_URL_CONTENT is empty` | Chưa cấu hình `.env` | Điền connection string trong `.env` |
| `No recommendations generated` | User chưa có interaction | Thử user ID khác hoặc seed thêm data |
| `ModuleNotFoundError` | Chưa cài dependencies | Chạy: `pip install -r requirements.txt` |

---

## 📄 License

GNU General Public License v3.0