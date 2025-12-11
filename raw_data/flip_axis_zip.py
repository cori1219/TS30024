import os
import re
import io
import zipfile
import pandas as pd


def read_csv_with_korean_support(raw_bytes):
    """
    CSV 인코딩을 utf-8 -> cp949 순으로 시도해서 DataFrame으로 리턴.
    실패하면 None 리턴.
    """
    for enc in ("utf-8", "cp949"):
        try:
            text = raw_bytes.decode(enc)
            if not text.strip():
                return None
            return pd.read_csv(io.StringIO(text))
        except Exception:
            continue
    return None


def flip_axis_in_zip(zip_path, axis_to_flip="x", output_zip_path=None,
                     only_when_xyz_present=True):
    """
    하나의 zip 파일 안에서 축 반전 수행.

    zip_path: 처리할 원본 zip 경로 (o_10.zip 같은 것)
    axis_to_flip: 'x' 또는 'y' 중 하나
    output_zip_path: 결과 zip 경로 (None이면 n+10 규칙으로 자동 생성)
    only_when_xyz_present: x,y,z 컬럼이 모두 있을 때만 반전할지 여부
    """
    zip_path = os.path.abspath(zip_path)
    dirname, basename = os.path.split(zip_path)

    # 출력 zip 이름 자동 생성 (o_10.zip -> o_20.zip)
    if output_zip_path is None:
        m = re.match(r"^(?P<prefix>.+)_(?P<num>\d+)\.zip$", basename)
        if m:
            prefix = m.group("prefix")
            num = int(m.group("num"))
            new_basename = f"{prefix}_{num + 10}.zip"
        else:
            stem, ext = os.path.splitext(basename)
            new_basename = f"{stem}_flipped{ext}"
        output_zip_path = os.path.join(dirname, new_basename)
    else:
        output_zip_path = os.path.abspath(output_zip_path)

    with zipfile.ZipFile(zip_path, "r") as zin, \
         zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:

        for info in zin.infolist():
            data = zin.read(info.filename)
            new_data = data

            # CSV만 처리
            if info.filename.lower().endswith(".csv"):
                df = read_csv_with_korean_support(data)
                if df is not None:
                    cols = set(df.columns)

                    # 지정한 축 컬럼이 있고, (옵션) x,y,z 다 있을 때만 반전
                    if axis_to_flip in cols and \
                       (not only_when_xyz_present or {"x", "y", "z"}.issubset(cols)):

                        df[axis_to_flip] = -df[axis_to_flip]
                        new_csv = df.to_csv(index=False)
                        new_data = new_csv.encode("utf-8")

            # 수정된 내용 쓰기
            zout.writestr(info, new_data)

    return output_zip_path


def process_all_zips(root_dir=".", axis_to_flip="x", prefixes=("o", "x")):
    """
    root_dir 아래를 전부 돌면서
    o_숫자.zip, x_숫자.zip 중 '원본'만 골라서
    o_(n+10).zip / x_(n+10).zip 을 만든다.

    - prefix_n.zip 옆에 prefix_(n-10).zip 이 있으면: 자식으로 보고 SKIP
      (예: o_20.zip 옆에 o_10.zip 있으면 o_20.zip은 생성된 파일이라 건들지 않음)
    - prefix_(n+10).zip 이 이미 있으면: 이미 처리된 걸로 보고 SKIP
    """
    root_dir = os.path.abspath(root_dir)
    print(f"root_dir: {root_dir}")

    pattern = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<num>\d+)\.zip$")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        fileset = set(filenames)

        for filename in filenames:
            m = pattern.match(filename)
            if not m:
                continue

            prefix = m.group("prefix")
            num = int(m.group("num"))

            # o, x 만 처리 (필요하면 여기 prefix 추가)
            if prefix not in prefixes:
                continue

            # 1) parent(prefix_(num-10).zip)가 있으면 => 이 파일은 "결과물"로 판단하고 SKIP
            parent_name = f"{prefix}_{num - 10}.zip"
            if num >= 10 and parent_name in fileset:
                print(f"[SKIP] generated file (has parent {parent_name}): "
                      f"{os.path.join(dirpath, filename)}")
                continue

            src_path = os.path.join(dirpath, filename)

            # 2) 이미 결과 파일(prefix_(num+10).zip)이 있으면 => 다시 안 만든다
            dst_basename = f"{prefix}_{num + 10}.zip"
            dst_path = os.path.join(dirpath, dst_basename)

            if dst_basename in fileset or os.path.exists(dst_path):
                print(f"[SKIP] output already exists: {dst_path}")
                continue

            # 3) 여기까지 오면 "진짜 원본"으로 보고 처리
            print(f"[PROC] {src_path} -> {dst_path}")
            new_path = flip_axis_in_zip(
                src_path,
                axis_to_flip=axis_to_flip,
                output_zip_path=dst_path,
                only_when_xyz_present=True,
            )
            print(f"[DONE] {new_path}")


if __name__ == "__main__":
    # 이 스크립트 파일 위치 기준
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 현재 구조가 .../TS30024/raw_data/o/flip_axis_zip.py 라고 했으니까
    # 한 단계 위(raw_data)를 root로 잡으면 o/, x/ 둘 다 훑는다.
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # 축을 y로 뒤집고 싶으면 axis_to_flip="y" 로 바꾸면 됨
    process_all_zips(
        root_dir=base_dir,
        axis_to_flip="x",      # "y"로 바꾸면 y축 반전
        prefixes=("o", "x"),   # 필요하면 다른 prefix 추가
    )

