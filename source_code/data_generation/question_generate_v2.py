import os
import csv
import base64
import mimetypes
import os.path
import random
import json
import time
from pathlib import Path
import google.generativeai as genai
import warnings
import argparse

warnings.filterwarnings("ignore")

# Danh sách API Keys


def create_prompt(class_name):
    if len(class_name) > 1:
        class_name = class_name.replace('_', ' ')

    prompt = f'''
        Đây là hình ảnh của món ăn tên là {class_name}.
        Hãy phân tích hình ảnh tạo ra các cặp câu hỏi - câu trả lời liên quan đến nội dung của bức ảnh.
        
        Chỉ được phép đặt các câu hỏi theo chủ đề sau đây:
            - Màu sắc chủ đạo của món ăn.
            - Món ăn là gì.
            - Có nước chấm hay không
            - Đây có phải là món nào đó không.
            
        Kết quả phải được trả về theo đúng định dạng: question,answer,question_type,answer_type
        
        Trong đó:
            - question_type gồm: color (hỏi về màu sắc như xanh, đỏ, vàng,...), regconition (hỏi món ăn là gì), yes/no (có nước chấm hay không)
            - answer_type gồm: text (dạng từ ví dụ như bánh táo, sushi, đỏ,...), boolean (có hoặc không)
        
        **Lưu ý:**
            - Kết quả trả về phải theo đúng định dạng, không chấp nhận các nội dung dư thừa, các kí tự markdown...
            - Không hỏi các hỏi nằm ngoài chủ đề đã đề cập.
            - Từ ngữ cần đơn giản và dễ hiểu.
            - Câu trả lời cần ngắn gọn, không chứa dấu câu hoặc kí tự đặc biệt.
            - Màu sắc chỉ lấy một màu (Ví dụ: vàng, đỏ,... không chấp nhận 'vàng nhạt', 'đỏ rực',...)
            - Câu hỏi liên quan đến tên thì trả lời như tên đã cung cấp (ví dụ: {class_name}, ramen).
            - question_type và answer_type phải được điền đúng như đã định nghĩa.
            - Ít nhất là 5 câu hỏi, thay đổi cách hỏi cho đa dạng 
            - Một chủ đề có thể hỏi nhiều câu nhưng theo cách khác nhau.
            - Câu trả lời cho màu sắc chỉ có các màu cơ bản sau: trắng, vàng, nâu, đỏ (không chấp nhận 'đỏ rực', 'vàng nhạt' hoặc 'vàng và đỏ',...).
            - Nhớ dịch tên món ăn sang tiếng Việt nếu cần thiết.
            
        Ví dụ bộ câu hỏi cho món {class_name}:
            Món này là món gì?,{class_name},recognition,text
            Đây là món gì?,{class_name},recognition,text
            Màu chủ đạo của món ăn là gì?,Vàng,color,text
            Món này có màu gì?,Vàng,color,text
            Có nước chấm không?,Có,yes/no,boolean
            Món này có nước chấm không?,Có,yes/no,boolean
        
    '''
    return prompt


def image_to_base64(image_path):
    """Chuyển đổi hình ảnh thành chuỗi base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
    return None


def get_mime_type(image_path):
    """Xác định MIME type của ảnh."""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type if mime_type else "image/jpeg"


def load_classes(csv_file):
    """Load dữ liệu từ classes.csv thành dictionary {label: class}"""
    class_map = {}
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_map[row['label']] = row['class']
    return class_map


def save_to_csv(output_file, data):
    """Lưu dữ liệu vào CSV."""
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(
                ["question", "answer", 'question_type', 'answer_type', "image_path"]
            )
        writer.writerows(data)


def load_progress(directory_path, label):
    """Đọc index ảnh đã xử lý từ file progress.json."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        return progress.get(directory_path, {}).get(label, 0)
    return 0  # Nếu chưa có, bắt đầu từ ảnh đầu tiên


def save_progress(directory_path, label, index):
    """Lưu index ảnh cuối cùng đã xử lý theo từng label."""
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)

    if directory_path not in progress:
        progress[directory_path] = {}

    progress[directory_path][label] = index

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=4)


API_KEYS = [
    "AIzaSyByKvNdPIMPvM1SkZuLROVjRpgXeMPlDj8",
    'AIzaSyDwEr0NXHknSpudmNOhXLFfRbDmjyQimts',
    'AIzaSyC0XiPeL_ZGZi9yiY5CWWNLkkVa_AZpa0c',
    'AIzaSyBkrEme6ADVznFoIK9SmcyTfAhVZhIXuaw',
    "AIzaSyBINMQKhJ86O_ktYk79mvcjlL8Y64Vn5P8",
    "AIzaSyCHyL6p90KCTpAPFc84YjA4Pc42unz3AtE",
    'AIzaSyAXKR4kVaH1THAHA-zlxsUnko1yAYdhrsc',
    'AIzaSyCwxb_2UsJjfgvdRjdK_34QxFYDaCqx94s',
    'AIzaSyA88H5ucQkwxohnEtFCb787fhmmqgioUfE',
    # Kiếu Thanh
    'AIzaSyDnQAL0Zqm4ALJfr8efL2v144PRBcFqoPM',
    'AIzaSyCBS4jvnwHKbNHCoA81fIiRN3L89BHLlAk',
    'AIzaSyAwYPL9pEsQOzvKmZiHa2qk3QZZgNkNokQ',
    'AIzaSyA7w3_6vGJ5aBlPmzJLoXN7ATG8SJJkjy4',
    # Yến Ngọc
    'AIzaSyCep0EG66ehSa8JM4K33E8dfjQYW4MrNq4',
    'AIzaSyApPz_ZB9V4YwcL3EFSt48fhBDTQH5P0Ac',
    'AIzaSyA53AM9Cd3wdpMlCC_FLpk-JLlhsaoj3u8',
    'AIzaSyDn_f1aeD5Svn5BasdVghIqX8nnagfOI8I',
    'AIzaSyB2TwxfuDVy7K2cYMtzbgGrgHlOiJojDXw',
    'AIzaSyDx7s6WaDrm3CBzUmYUAp0RNoWpIlkYhJk',
    'AIzaSyDDxZPw3ZFPSMferltdAlgr_78j7RFwCFk',
    'AIzaSyDnpuES_OmtS3DZVu8nkq3uBop_gNznSSA',
    'AIzaSyB3eX8o6iAT8HwKGh5RILRxyeWt5wL9FYY',
    'AIzaSyADhraGGWwOkHZsBZQlv8f7TzDt_4RjYQg',
    'AIzaSyC-eTKRCJrQ1XfbjvQ7dcjLXKKRCNFSeYc',
    'AIzaSyBSey4pTqFOCRGsEtH-6K5bwzmfzmJRSmU',
    'AIzaSyBdRr2dBb3MgjY8vRwGg-P0q2yDw156iT4',
    # Thanh Phong
    'AIzaSyAAMN8FJ4rU485nVwL7r08KUMyq2yU4_28',
    'AIzaSyDtlwSXsQecoCLzaTYfyDTWnCNFWcydAKs',
    # Quang Trung
    'AIzaSyAj5VaV099t_urdHR5q2WcxnEmFF5nKJzw',
    'AIzaSyAZkNw1vptUAxEsdae4JZ1FgsP_EA4jQ_M',
    'AIzaSyCF1SESd16huPRT9pW3FlrdJpI61h_QslU',
    'AIzaSyCEMIzCsA8O8DqGxgGuZcOzUWtHHmWlNIY',
    'AIzaSyBBB9VJOHJDq7PqdsHnYFsoqCP1REBuvKA',
    'AIzaSyA_OOgezh4lIUc83GhxT9o1TW-Ru6XyyeU',
    'AIzaSyDszAn3dIiWHWCdqqu4Jg9w9gFg1XuKNJs',
    'AIzaSyAwW2-zJQugNu6vLXE4S-eMaq4myPJhJ84',
]


def analyze_image(model, image_path, class_name, timeout=10):
    """Gửi ảnh lên API Gemini để lấy câu hỏi & câu trả lời, với timeout."""
    base64_image = image_to_base64(image_path)
    if not base64_image:
        raise Exception("Không thể chuyển ảnh sang base64")

    contents = [
        create_prompt(class_name),
        {"mime_type": get_mime_type(image_path), "data": base64_image},
    ]

    print('Processing...')
    response = model.generate_content(contents)
    text = response.text
    lines = text.split('\n')
    return [line.split(',') for line in lines if line]


def process_images_in_directory(root, folder, output_csv, allowed_labels=None):
    """Duyệt qua ảnh trong thư mục và xử lý từng ảnh, có khả năng tiếp tục từ ảnh chưa xử lý.

    - `allowed_labels`: Danh sách label được phép xử lý (list). Nếu None, xử lý tất cả.
    """
    directory_path = os.path.join(root, folder)

    api_key = random.choice(API_KEYS)
    print('###### Using API key:', api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Duyệt qua từng thư mục con (label)
    label_folders = [p for p in Path(directory_path).iterdir() if p.is_dir()]

    elapsed_time = 0
    running_time = 0

    for label_folder in label_folders:
        label = label_folder.name

        # Nếu có danh sách allowed_labels, chỉ xử lý label nằm trong danh sách đó
        if allowed_labels is None:
            print(f'Không có allowed_labels')
        if allowed_labels is not None and label not in allowed_labels:
            print(f"Bỏ qua label: {label}")
            continue

        image_files = sorted(
            label_folder.glob("*.jpg"),
            key=lambda x: int(x.stem.split('_')[-1].replace('.jpg', '')),
        )

        start_index = load_progress(directory_path, label)

        print(
            f"\nLabel '{label}': Có {len(image_files)} ảnh. Bắt đầu từ ảnh số {start_index}."
        )

        for i in range(start_index, len(image_files)):
            start = time.time()
            image_file = image_files[i]
            image_name = image_file.name
            print()
            print('-' * 60)
            print(f">>> {image_file} ({i+1}/{len(image_files)}) <<<")

            try:
                results = analyze_image(model, image_file, label)
                data = [
                    [
                        item[0],
                        item[1],
                        item[2],
                        item[3],
                        f'{folder}/{label}/{image_name}',
                    ]
                    for item in results
                ]

                for line in data:
                    print(line)

                save_to_csv(output_csv, data)  # Lưu kết quả
                save_progress(directory_path, label, i + 1)  # Lưu tiến trình

                time.sleep(7)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_name}: {e}")
                time.sleep(2)  # Đợi 2 giây rồi tiếp tục ảnh tiếp theo
                continue  # Không gọi lại `process_images_in_directory()`, tránh đệ quy vô hạn

            end = time.time()
            elapsed_time += end - start
            running_time += end - start

            if i % 5 == 0:
                print()
                print(
                    f">>>>>>>>>>>>>>> ETA: {(elapsed_time / 5 * (len(image_files) - i) / 60):.2f} mins, Elapsed time: {running_time / 60:.2f} mins <<<<<<<<<<<<<<<"
                )

                # Đổi API key sau mỗi 5 ảnh
                api_key = random.choice(API_KEYS)
                print('###### Using API key:', api_key)
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                elapsed_time = 0


# Đường dẫn thư mục ảnh
DATA_DIR = "annotations"
os.makedirs(DATA_DIR, exist_ok=True)

# Khởi tạo trình phân tích cú pháp đối số
parser = argparse.ArgumentParser(
    description='Generate questions and answers for images in specified directories.'
)
parser.add_argument('--train', action='store_true', help='Process training dataset.')
parser.add_argument(
    '--validation', action='store_true', help='Process validation dataset.'
)
parser.add_argument('--test', action='store_true', help='Process test dataset.')
parser.add_argument(
    '--progress', type=str, default='progress.json', help='Name of the progress file.'
)
parser.add_argument(
    '--labels',
    nargs='+',
    type=str,
    help='Specific labels to process (e.g., apple banana). If not specified, all labels will be processed.',
)

parser.add_argument(
    '--output',
    type=str,
    default='output.csv',
    help='Name of the output CSV file.',
)

# Phân tích các đối số dòng lệnh
args = parser.parse_args()

labels = args.labels
print(labels)


PROGRESS_FILE = args.progress + '.json'
# Xử lý dữ liệu huấn luyện nếu được chỉ định
output_csv = args.output
if args.train:
    DATA_DIR = "annotations"
    os.makedirs(DATA_DIR, exist_ok=True)
    BASE_IMAGE_DIR = os.path.join('images')
    TRAIN_DIR = os.path.join(BASE_IMAGE_DIR, "train")
    TRAIN_CSV = os.path.join(DATA_DIR, output_csv)
    process_images_in_directory(
        root="images", folder="train", output_csv=output_csv, allowed_labels=labels
    )

# Xử lý dữ liệu xác thực nếu được chỉ định
if args.validation:
    DATA_DIR = "annotations"
    os.makedirs(DATA_DIR, exist_ok=True)
    BASE_IMAGE_DIR = os.path.join('images')
    VAL_DIR = os.path.join(BASE_IMAGE_DIR, "validation")
    process_images_in_directory(
        root="images", folder="validation", output_csv=output_csv, allowed_labels=labels
    )

# Xử lý dữ liệu kiểm tra nếu được chỉ định
if args.test:
    DATA_DIR = "annotations"
    os.makedirs(DATA_DIR, exist_ok=True)
    BASE_IMAGE_DIR = os.path.join('images')
    TEST_DIR = os.path.join(BASE_IMAGE_DIR, "test")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    process_images_in_directory(
        root="images", folder="test", output_csv=output_csv, allowed_labels=labels
    )
